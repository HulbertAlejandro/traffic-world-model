"""v2 control phase, checked before training any controller (docs/v2/ADDENDUM_CONTROL.md, Paso 3).

a) Bridge: replaying a recorded episode in the real corridor gives the controller exactly the
   dataset's normalized states (v1's bridge bug: a different normalization than training).
b) The Dream Environment never calls SUMO.
c) PPO's joint action reaches the model as encode_actions' one-hot.
d) Two scaled corridor environments in one process each read their own simulation (C1).
e) Every arm uses the same PPO configuration object, changing only the addendum's fields.
f) The training script refuses to write into official or world-model folders.
"""

import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import ControllerConfig  # noqa: E402
from datasets.latent_sequence_dataset import encode_actions  # noqa: E402
from training import train_controller_v2 as tc  # noqa: E402
from training.v2_compression_experiment import build_temporal_model  # noqa: E402

RAW_VALIDATION = ROOT_DIR / "datasets" / "v2" / "raw" / "validation"
PROCESSED_VALIDATION = ROOT_DIR / "datasets" / "v2" / "processed" / "validation.npz"
BRIDGE_EPISODE = 21001  # a fijo_2_3 episode of the validation split


def _tiny_world_model(tmp_path: Path, architecture: str = "lstm") -> tuple[Path, Path]:
    """A small untrained world model and a synthetic seed-episode file, same formats as the real ones."""
    hp = {"architecture": architecture, "latent_dim": 104, "action_dim": 8, "sequence_length": 16}
    hp |= {"hidden_dim": 8} if architecture == "lstm" else {"d_model": 8, "nhead": 2, "num_layers": 1,
                                                             "dim_feedforward": 16, "dropout": 0.0}
    model_dir = tmp_path / "wm"
    model_dir.mkdir()
    torch.manual_seed(0)
    torch.save(build_temporal_model(hp).state_dict(), model_dir / "world_model_best.pt")
    (model_dir / "world_model_best.json").write_text(json.dumps(hp), encoding="utf-8")
    (model_dir / "reward_scaler.json").write_text(json.dumps({"reward_mean": -45.0, "reward_std": 70.0}),
                                                  encoding="utf-8")
    rng = np.random.default_rng(0)
    T, n_ep = 20, 2
    episodes = tmp_path / "seed_episodes.npz"
    np.savez(episodes, z=rng.normal(size=(T * n_ep, 104)).astype(np.float32),
             next_z=rng.normal(size=(T * n_ep, 104)).astype(np.float32),
             actions=rng.integers(0, 2, size=(T * n_ep, 4)), rewards=-rng.random(T * n_ep).astype(np.float32),
             episode_id=np.repeat([7, 8], T), time_step=np.tile(np.arange(T), n_ep))
    return model_dir, episodes


# --------------------------------------------------------------------------- a) bridge
@pytest.mark.skipif(not (RAW_VALIDATION / f"episode_{BRIDGE_EPISODE}.npz").exists() or not PROCESSED_VALIDATION.exists(),
                    reason="v2 dataset not present (not committed)")
def test_bridge_reproduces_the_datasets_normalized_states():
    from environments.scaled_corridor_environment import ScaledCorridorEnvironment

    with np.load(RAW_VALIDATION / f"episode_{BRIDGE_EPISODE}.npz") as raw:
        actions, raw_rewards = raw["actions"], raw["rewards"]
    with np.load(PROCESSED_VALIDATION) as d:
        mask = d["episode_id"] == BRIDGE_EPISODE
        order = np.argsort(d["time_step"][mask])
        states, next_states = d["states"][mask][order], d["next_states"][mask][order]
    env = ScaledCorridorEnvironment()
    try:
        observed = [env.reset(seed=BRIDGE_EPISODE)[0]]
        rewards = []
        for action in actions:
            obs, reward, *_ = env.step(action)
            observed.append(obs)
            rewards.append(reward)
    finally:
        env.close()
    observed = np.array(observed)
    assert observed.shape == (61, 104)
    np.testing.assert_allclose(observed[:-1], states, rtol=0, atol=1e-5)
    np.testing.assert_allclose(observed[1:], next_states, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(np.array(rewards, dtype=np.float32), raw_rewards)


# --------------------------------------------------------------------------- b) no SUMO in the dream
@pytest.mark.parametrize("architecture", ["lstm", "transformer"])
def test_dream_environment_never_calls_sumo(tmp_path, monkeypatch, architecture):
    import sumo_rl
    import traci

    from environments import corridor_environment
    from environments.corridor_dream_environment import CorridorDreamEnvironment

    def forbidden(*args, **kwargs):
        raise AssertionError("the Dream Environment called SUMO")

    for target, name in ((traci, "start"), (traci, "connect"), (traci, "init"), (traci, "getConnection"),
                         (sumo_rl.SumoEnvironment, "__init__"),
                         (corridor_environment.CorridorTrafficEnvironment, "__init__")):
        monkeypatch.setattr(target, name, forbidden)

    model_dir, episodes = _tiny_world_model(tmp_path, architecture)
    dream = CorridorDreamEnvironment(model_dir, seed_episodes_path=episodes, max_dream_steps=7, eval_seed=0)
    rng = np.random.default_rng(1)
    for _ in range(3):
        obs, info = dream.reset()
        assert obs.shape == (104,) and info["seeded_from_real_data"]
        truncated, steps = False, 0
        while not truncated:
            obs, reward, terminated, truncated, info = dream.step(rng.integers(0, 2, size=4))
            steps += 1
            assert obs.shape == (104,) and not terminated and -345.43 <= reward <= 0.0
        assert steps == 7


def test_dream_reset_with_options_seeds_the_recorded_window(tmp_path):
    from environments.corridor_dream_environment import CorridorDreamEnvironment

    model_dir, episodes = _tiny_world_model(tmp_path)
    dream = CorridorDreamEnvironment(model_dir, seed_episodes_path=episodes)
    obs, info = dream.reset(options={"episode_id": 8, "start": 3})
    with np.load(episodes) as d:
        z = d["z"][d["episode_id"] == 8]
        acts = d["actions"][d["episode_id"] == 8]
    np.testing.assert_array_equal(obs, z[3 + 15])
    np.testing.assert_array_equal(dream._window_actions.numpy(), encode_actions(torch.from_numpy(acts[3:19]), 8).numpy())
    with pytest.raises(ValueError):
        dream.reset(options={"episode_id": 8, "start": 5})  # 5 + 16 > 20


# --------------------------------------------------------------------------- c) action translation
def test_ppo_action_reaches_the_model_as_encode_actions(tmp_path, monkeypatch):
    from environments import corridor_dream_environment as cde

    model_dir, episodes = _tiny_world_model(tmp_path)
    dream = cde.CorridorDreamEnvironment(model_dir, seed_episodes_path=episodes, eval_seed=0)
    assert dream.action_space.nvec.tolist() == [2, 2, 2, 2]
    captured = []
    real_predict = cde.predict_next_step

    def spy(model, window_z, window_actions, *args):
        captured.append(window_actions.clone())
        return real_predict(model, window_z, window_actions, *args)

    monkeypatch.setattr(cde, "predict_next_step", spy)
    for bits in itertools.product((0, 1), repeat=4):
        action = np.array(bits, dtype=np.int64)
        expected = encode_actions(torch.from_numpy(action[None]), 8)[0]
        torch.testing.assert_close(dream.encode_action(action), expected)
        dream.reset()
        history = dream._window_actions.clone()
        dream.step(action)
        torch.testing.assert_close(captured[-1][-1], expected)          # the hypothetical action, last row
        torch.testing.assert_close(captured[-1][:-1], history[:-1])     # the recorded ones, untouched
    # Layout: per signal [keep, switch], A0..D0 in order.
    torch.testing.assert_close(dream.encode_action([1, 0, 0, 1]), torch.tensor([0., 1., 1., 0., 1., 0., 0., 1.]))


@pytest.mark.parametrize("architecture", ["lstm", "transformer"])
def test_real_world_models_load_in_the_dream(architecture):
    from environments.corridor_dream_environment import CorridorDreamEnvironment, world_model_dir

    model_dir = world_model_dir(architecture, 0)
    if not (model_dir / "world_model_best.pt").exists():
        pytest.skip("v2 world-model weights not present (not committed)")
    dream = CorridorDreamEnvironment(model_dir, eval_seed=0)
    assert dream.architecture == architecture and dream.sequence_length == 16 and dream.action_dim == 8
    obs, _ = dream.reset()
    obs, reward, *_ = dream.step(np.zeros(4, dtype=np.int64))
    assert obs.shape == (104,) and np.isfinite(obs).all() and np.isfinite(reward)


# --------------------------------------------------------------------------- d) concurrent connections
def test_two_concurrent_scaled_corridor_environments_each_read_their_own_simulation():
    from environments.scaled_corridor_environment import ScaledCorridorEnvironment

    seeds = (19900, 19901)
    rng = np.random.default_rng(0)
    actions = [rng.integers(0, 2, size=(8, 4)) for _ in seeds]

    def solo(seed, acts):
        env = ScaledCorridorEnvironment()
        try:
            out = [env.reset(seed=seed)[0]]
            out += [env.step(a)[0] for a in acts]
        finally:
            env.close()
        return np.array(out)

    solo_runs = [solo(s, a) for s, a in zip(seeds, actions)]
    envs = [ScaledCorridorEnvironment(), ScaledCorridorEnvironment()]
    try:
        runs = [[env.reset(seed=s)[0]] for env, s in zip(envs, seeds)]  # the second started last
        for step in range(8):
            for k, env in enumerate(envs):
                runs[k].append(env.step(actions[k][step])[0])
    finally:
        for env in envs:
            env.close()
    assert not np.array_equal(solo_runs[0], solo_runs[1])
    for k in range(2):
        np.testing.assert_array_equal(np.array(runs[k]), solo_runs[k])


# --------------------------------------------------------------------------- e) same PPO configuration
def _changed_fields(config: ControllerConfig, base: ControllerConfig) -> set[str]:
    return {k for k, v in asdict(config).items() if asdict(base)[k] != v}


def test_every_arm_uses_the_same_ppo_configuration():
    base = ControllerConfig()
    for seed in range(10):
        lstm, transformer = tc.ppo_config("dream_lstm", seed), tc.ppo_config("dream_transformer", seed)
        assert lstm == transformer
        assert tc.ppo_kwargs(lstm) == tc.ppo_kwargs(transformer)
        assert _changed_fields(lstm, base) <= {"seed"} and lstm.seed == seed
        assert lstm.dream_max_steps == 7 and lstm.total_timesteps == 50_000 and not lstm.normalize_obs
        for budget in (10_000, 30_000):
            direct = tc.ppo_config("direct", seed, budget)
            assert _changed_fields(direct, lstm) == {"total_timesteps", "normalize_obs"}
            assert direct.normalize_obs and direct.total_timesteps == budget
            kwargs_direct, kwargs_dream = tc.ppo_kwargs(direct), tc.ppo_kwargs(lstm)
            assert kwargs_direct == kwargs_dream
    with pytest.raises(ValueError):
        tc.ppo_config("direct", 0)  # the direct budget is never implicit
    assert tc.eval_freq("dream_lstm", base) == tc.eval_freq("dream_transformer", base) == 5000
    assert not set(tc.SELECTION_SEEDS) & set(tc.TRAIN_SPLIT_SEEDS)
    assert tc.DIRECT_TRAINING_SEED_START >= 30000


def test_dream_arms_use_the_world_model_of_their_own_seed_and_architecture():
    from environments.corridor_dream_environment import world_model_dir

    assert world_model_dir("lstm", 3).name == "lstm_raw_s3"
    assert world_model_dir("transformer", 7).name == "transformer_raw_s7"
    assert tc.DREAM_ARMS == {"dream_lstm": "lstm", "dream_transformer": "transformer"}


# --------------------------------------------------------------------------- f) output guard
def test_training_refuses_official_and_world_model_folders(tmp_path):
    protected = [*tc.PROTECTED_DIRS, *tc.PROTECTED_ROOTS,
                 tc.CHECKPOINT_DIR / "v2" / "arch_comparison" / "lstm_raw_s0",
                 tc.CHECKPOINT_DIR / "v2" / "arch_comparison" / "new_subfolder"]
    for folder in protected:
        with pytest.raises(SystemExit):
            tc.check_v2_output_dir(folder, overwrite=True, overwrite_official=False)
    # The whole script stops before building anything.
    with pytest.raises(SystemExit):
        tc.main(["--arm", "dream_lstm", "--seed", "0", "--output-dir",
                 str(tc.CHECKPOINT_DIR / "v2" / "arch_comparison" / "lstm_raw_s0")])
    # An existing result is not overwritten without --overwrite; an empty folder is fine.
    (tmp_path / "run_info.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        tc.check_v2_output_dir(tmp_path, overwrite=False, overwrite_official=False)
    assert tc.check_v2_output_dir(tmp_path, overwrite=True, overwrite_official=False) == tmp_path.resolve()
    assert tc.check_v2_output_dir(tmp_path / "empty", overwrite=False, overwrite_official=False)


def test_evaluation_refuses_train_seeds_and_unconfirmed_held_out_splits():
    from scripts.v2 import evaluate_control_v2 as ev

    assert ev.scenario_seeds("validation", None, False) == list(range(21000, 21024))
    for bad in ([20000], [20111], [22000]):
        with pytest.raises(SystemExit):
            ev.scenario_seeds("validation", bad, False)
    for split in ("test", "ood"):
        with pytest.raises(SystemExit):
            ev.scenario_seeds(split, None, False)
    assert ev.scenario_seeds("test", None, True) == list(range(22000, 22024))


def test_reference_agreement_only_reads_the_simulation():
    """Asking the five references for their action at every step must not change the trajectory,
    and a reference must agree with itself on every step and signal."""
    from environments.scaled_corridor_environment import ScaledCorridorEnvironment
    from scripts.v2 import evaluate_control_v2 as ev
    from scripts.v2.corridor_policies import REFERENCE_POLICIES, make_reference_policy

    env = ScaledCorridorEnvironment()
    try:
        drive = make_reference_policy("espera_mas_larga")
        act = lambda state, step: drive(env.env, step)  # noqa: E731
        plain = ev.run_episode(env.env, act, 21005)
        shadow = {name: make_reference_policy(name) for name in REFERENCE_POLICIES}
        with_shadow = ev.run_episode(env.env, act, 21005, shadow)
    finally:
        env.close()
    for key in ("reward", "reward_A0", "reward_B0", "reward_C0", "reward_D0", "switches_A0", "switches_C0"):
        assert with_shadow[key] == plain[key]
    assert all(with_shadow[f"agree_espera_mas_larga_{ts}"] == 1.0 for ts in ("A0", "B0", "C0", "D0"))
    assert "agree_espera_mas_larga_A0" not in plain
    assert any(with_shadow[f"agree_max_presion_{ts}"] < 1.0 for ts in ("A0", "B0", "C0", "D0"))
