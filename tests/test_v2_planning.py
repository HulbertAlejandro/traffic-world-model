"""v2.1-A planner, checked before simulating anything (docs/v2/ADDENDUM_PLANIFICACION.md, Paso 3).

a) Planning never calls SUMO (only the caller applies the chosen action).
b) Fed the recorded actions, the planner's imagined returns match Experiment 1's rollout_episode.
c) Bridge: the planner receives the same normalized states as the dream PPO.
d) Tie-break and start-of-episode padding behave as pre-registered.
e) Same seeds, same result.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.latent_sequence_dataset import encode_actions  # noqa: E402
from environments.corridor_planner import (  # noqa: E402
    JOINT_ACTIONS,
    CorridorPlanner,
    choose,
    imagined_returns,
    keep_continuation,
    ppo_continuation,
)
from training.v2_compression_experiment import build_temporal_model  # noqa: E402

RAW_VALIDATION = ROOT_DIR / "datasets" / "v2" / "raw" / "validation"
PROCESSED_VALIDATION = ROOT_DIR / "datasets" / "v2" / "processed" / "validation.npz"
REAL_LSTM = ROOT_DIR / "models" / "checkpoints" / "v2" / "arch_comparison" / "lstm_raw_s0"


def _tiny_world_model(tmp_path: Path, architecture: str = "lstm", T: int = 30, zero: bool = False):
    hp = {"architecture": architecture, "latent_dim": 104, "action_dim": 8, "sequence_length": 16}
    hp |= {"hidden_dim": 8} if architecture == "lstm" else {"d_model": 8, "nhead": 2, "num_layers": 1,
                                                             "dim_feedforward": 16, "dropout": 0.0}
    model_dir = tmp_path / f"wm_{architecture}"
    model_dir.mkdir()
    torch.manual_seed(0)
    model = build_temporal_model(hp)
    if zero:  # constant prediction: every action scores the same
        for p in model.parameters():
            torch.nn.init.zeros_(p)
    torch.save(model.state_dict(), model_dir / "world_model_best.pt")
    (model_dir / "world_model_best.json").write_text(json.dumps(hp), encoding="utf-8")
    (model_dir / "reward_scaler.json").write_text(json.dumps({"reward_mean": -45.0, "reward_std": 70.0}),
                                                  encoding="utf-8")
    rng = np.random.default_rng(0)
    episodes = tmp_path / f"episodes_{architecture}.npz"
    np.savez(episodes, z=rng.normal(size=(2 * T, 104)).astype(np.float32),
             next_z=rng.normal(size=(2 * T, 104)).astype(np.float32),
             actions=rng.integers(0, 2, size=(2 * T, 4)), rewards=-100 * rng.random(2 * T).astype(np.float32),
             episode_id=np.repeat([7, 8], T), time_step=np.tile(np.arange(T), 2))
    return model_dir, episodes


# --------------------------------------------------------------------------- a) no SUMO
@pytest.mark.parametrize("continuation", ["solo", "ppo"])
def test_planning_never_calls_sumo(tmp_path, monkeypatch, continuation):
    import sumo_rl
    import traci
    from stable_baselines3 import PPO

    from environments import corridor_environment
    from environments.corridor_dream_environment import CorridorDreamEnvironment

    model_dir, episodes = _tiny_world_model(tmp_path)
    cont = keep_continuation
    if continuation == "ppo":
        cont = ppo_continuation(PPO("MlpPolicy", CorridorDreamEnvironment(model_dir, seed_episodes_path=episodes),
                                    seed=0, device="cpu"))
    planner = CorridorPlanner(model_dir, horizon=5, continuation=cont)

    def forbidden(*args, **kwargs):
        raise AssertionError("planning called SUMO")

    for target, name in ((traci, "start"), (traci, "connect"), (traci, "init"), (traci, "getConnection"),
                         (traci, "simulationStep"), (sumo_rl.SumoEnvironment, "__init__"),
                         (sumo_rl.SumoEnvironment, "step"), (corridor_environment.CorridorTrafficEnvironment, "__init__"),
                         (corridor_environment.CorridorTrafficEnvironment, "step")):
        monkeypatch.setattr(target, name, forbidden)
    rng = np.random.default_rng(1)
    for _ in range(20):
        action = planner.act(rng.normal(size=104).astype(np.float32))
        assert action.shape == (4,) and set(np.unique(action)) <= {0, 1}
    assert planner.last_returns.shape == (16,)


# --------------------------------------------------------------------------- b) same returns as Experiment 1
def _rollout_equivalence(model_dir, episodes):
    """Fed the recorded actions, the planner's imagined returns equal those of Experiment 1's
    rollout_episode (actions aligned by time index; ADDENDUM_PLANIFICACION.md, section 11)."""
    from environments.corridor_dream_environment import load_temporal_model
    from evaluation.world_model_evaluation import load_episodes
    from scripts.v2.control_fidelity import HORIZON, rollout_episode_returns

    model, hp, scaler = load_temporal_model(model_dir)
    L, checked = hp["sequence_length"], 0
    for ep in load_episodes(episodes).values():
        expected = rollout_episode_returns(model, scaler, ep, L, HORIZON)
        for start, ref in enumerate(expected):
            window_z = torch.from_numpy(ep["z"][start:start + L])
            window_a = encode_actions(torch.from_numpy(ep["actions"][start:start + L]), 8)
            recorded = ep["actions"][start + L - 1: start + L - 1 + HORIZON]
            replay = lambda h, z, rec=recorded: np.repeat(rec[h + 1][None], len(z), axis=0)  # noqa: E731
            got = imagined_returns(model, scaler["reward_mean"], scaler["reward_std"], window_z, window_a,
                                   recorded[:1], replay, HORIZON)
            assert got[0] == pytest.approx(ref, abs=1e-2)
            checked += 1
    return checked


@pytest.mark.parametrize("architecture", ["lstm", "transformer"])
def test_imagined_returns_match_rollout_episode(tmp_path, architecture):
    model_dir, episodes = _tiny_world_model(tmp_path, architecture)
    assert _rollout_equivalence(model_dir, episodes) == 2 * (30 - 16 - 7 + 1)


@pytest.mark.parametrize("architecture", ["lstm", "transformer"])
def test_imagined_returns_match_rollout_episode_with_real_models(tmp_path, architecture):
    from environments.corridor_dream_environment import SEQUENCE_DATA_DIR, world_model_dir

    model_dir = world_model_dir(architecture, 0)
    if not (model_dir / "world_model_best.pt").exists() or not (SEQUENCE_DATA_DIR / "validation_seq.npz").exists():
        pytest.skip("v2 world models / sequence data not present (not committed)")
    with np.load(SEQUENCE_DATA_DIR / "validation_seq.npz") as d:  # one validation episode is enough
        mask = d["episode_id"] == 21000
        one = tmp_path / "one_episode.npz"
        np.savez(one, **{k: d[k][mask] for k in d.files})
    assert _rollout_equivalence(model_dir, one) == 60 - 16 - 7 + 1


# --------------------------------------------------------------------------- c) bridge
@pytest.mark.skipif(not (RAW_VALIDATION / "episode_21001.npz").exists() or not PROCESSED_VALIDATION.exists()
                    or not (REAL_LSTM / "world_model_best.pt").exists(), reason="v2 data / weights not present")
def test_planner_receives_the_same_normalized_states_as_the_dream_ppo(monkeypatch):
    from environments.scaled_corridor_environment import ScaledCorridorEnvironment
    from scripts.v2 import evaluate_control_v2 as ev

    with np.load(RAW_VALIDATION / "episode_21001.npz") as raw:
        actions = raw["actions"]
    with np.load(PROCESSED_VALIDATION) as d:
        mask = d["episode_id"] == 21001
        states = d["states"][mask][np.argsort(d["time_step"][mask])]
    received = []

    def spy(self, state):  # records what the planner gets and replays the dataset's actions
        received.append(np.array(state))
        return actions[len(received) - 1]

    monkeypatch.setattr(CorridorPlanner, "act", spy)
    env = ScaledCorridorEnvironment()
    try:
        policy = ev.parse_policy("p=plan:lstm,solo,3,0")
        [(label, fn)] = ev.build_action_functions(policy, env)
        ev.run_episode(env.env, fn, 21001)
    finally:
        env.close()
    assert label == "lstm_s0" and len(received) == 60
    np.testing.assert_allclose(np.array(received), states, rtol=0, atol=1e-5)


# --------------------------------------------------------------------------- d) tie-break and padding
def test_tie_break_prefers_fewer_switches_then_the_fixed_order():
    keep = int(np.flatnonzero((JOINT_ACTIONS == 0).all(axis=1))[0])
    assert choose(np.zeros(16)) == keep                       # all tied: keep the four phases
    returns = np.full(16, -10.0)
    one_a0 = int(np.flatnonzero((JOINT_ACTIONS == [1, 0, 0, 0]).all(axis=1))[0])
    one_d0 = int(np.flatnonzero((JOINT_ACTIONS == [0, 0, 0, 1]).all(axis=1))[0])
    two = int(np.flatnonzero((JOINT_ACTIONS == [1, 1, 0, 0]).all(axis=1))[0])
    returns[[one_a0, one_d0, two]] = -1.0
    assert choose(returns) == one_d0                          # fewest switches; first in product order
    returns[keep] = -1.0
    assert choose(returns) == keep
    returns[two] = 0.0
    assert choose(returns) == two                             # a strictly better return always wins


def test_constant_model_keeps_every_phase(tmp_path):
    model_dir, _ = _tiny_world_model(tmp_path, zero=True)
    planner = CorridorPlanner(model_dir, horizon=3)
    for _ in range(5):
        np.testing.assert_array_equal(planner.act(np.ones(104, dtype=np.float32)), [0, 0, 0, 0])
    assert np.unique(planner.last_returns).size == 1


def test_start_of_episode_window_is_padded_with_the_initial_state_and_keep(tmp_path):
    model_dir, _ = _tiny_world_model(tmp_path)
    planner = CorridorPlanner(model_dir, horizon=3)
    states = [np.full(104, float(i), dtype=np.float32) for i in range(20)]
    planner.act(states[0])
    z, a = planner.window()
    np.testing.assert_array_equal(z.numpy(), np.stack([states[0]] * 16))
    np.testing.assert_array_equal(a[:15].numpy(), encode_actions(torch.zeros(15, 4, dtype=torch.long), 8).numpy())
    for s in states[1:4]:
        planner.act(s)
    z, a = planner.window()  # t = 3: 12 copies of s_0, then s_0..s_3
    np.testing.assert_array_equal(z.numpy(), np.stack([states[0]] * 12 + states[0:4]))
    applied = np.array(planner._actions)
    expected = np.concatenate([np.zeros((12, 4), dtype=np.int64), applied[:3], np.zeros((1, 4), dtype=np.int64)])
    np.testing.assert_array_equal(a.numpy(), encode_actions(torch.from_numpy(expected), 8).numpy())
    for s in states[4:20]:
        planner.act(s)
    z, a = planner.window()  # t = 19: the last 16 real states, no padding
    np.testing.assert_array_equal(z.numpy(), np.stack(states[4:20]))
    np.testing.assert_array_equal(a[:15].numpy(),
                                  encode_actions(torch.from_numpy(np.array(planner._actions[4:19])), 8).numpy())


# --------------------------------------------------------------------------- e) determinism
def test_same_inputs_same_plans(tmp_path):
    model_dir, _ = _tiny_world_model(tmp_path)
    rng = np.random.default_rng(3)
    states = rng.normal(size=(25, 104)).astype(np.float32)
    runs = []
    for _ in range(2):
        planner = CorridorPlanner(model_dir, horizon=5)
        actions, returns = [], []
        for s in states:
            actions.append(planner.act(s))
            returns.append(planner.last_returns.copy())
        runs.append((np.array(actions), np.array(returns)))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


@pytest.mark.skipif(not (REAL_LSTM / "world_model_best.pt").exists(), reason="v2 world-model weights not present")
def test_same_seed_same_episode_in_sumo():
    from environments.scaled_corridor_environment import ScaledCorridorEnvironment
    from scripts.v2 import evaluate_control_v2 as ev

    env = ScaledCorridorEnvironment()
    try:
        records = []
        for _ in range(2):
            [(_, fn)] = ev.build_action_functions(ev.parse_policy("p=plan:lstm,solo,3,0"), env)
            records.append(ev.run_episode(env.env, fn, 19900))  # a tests-only seed, outside every split
    finally:
        env.close()
    for key in ("reward", "reward_A0", "reward_B0", "reward_C0", "reward_D0",
                "switches_A0", "switches_B0", "switches_C0", "switches_D0"):
        assert records[0][key] == records[1][key]


def test_evaluation_refuses_held_out_v21_test_and_phase3_results(tmp_path):
    from scripts.v2 import evaluate_control_v2 as ev

    assert ev.scenario_seeds("validation_v21", None, False) == list(range(24000, 24024))
    with pytest.raises(SystemExit):
        ev.scenario_seeds("test_v21", None, False)
    with pytest.raises(SystemExit):
        ev.scenario_seeds("validation_v21", [22000], False)
    with pytest.raises(SystemExit):
        ev.check_output(ev.PHASE3_RESULTS_DIR / "anything", overwrite=True, overwrite_official=False)
    (tmp_path / "out.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        ev.check_output(tmp_path / "out", overwrite=False, overwrite_official=False)
    ev.check_output(tmp_path / "out", overwrite=True, overwrite_official=False)
    ev.check_output(tmp_path / "new", overwrite=False, overwrite_official=False)
