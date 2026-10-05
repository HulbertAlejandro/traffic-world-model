"""Window alignment of the Dream Environment (docs/v2/ADDENDUM_PLANIFICACION.md, section 11).

v2, CorridorDreamEnvironment(window_alignment=...):
a) "aligned" with the dataset's actions gives Experiment 1's rollout_episode returns.
b) "legacy" (the default) reproduces the published stage-1 fidelity (docs/results/v2/control/
   fidelity_validation.json) exactly, and differs from rollout_episode (the defect).
c) Both share the interface and neither calls SUMO.

v1 (checked only, no v1 file changed): its DreamEnvironment has the same legacy shift.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.corridor_dream_environment import (  # noqa: E402
    SEQUENCE_DATA_DIR,
    CorridorDreamEnvironment,
    load_temporal_model,
    world_model_dir,
)
from evaluation.world_model_evaluation import load_episodes  # noqa: E402
from scripts.v2.control_fidelity import HORIZON, imagined_vs_real, rollout_episode_returns  # noqa: E402
from training.v2_compression_experiment import build_temporal_model  # noqa: E402

PUBLISHED_FIDELITY = ROOT_DIR / "docs" / "results" / "v2" / "control" / "fidelity_validation.json"
VALIDATION_SEQ = SEQUENCE_DATA_DIR / "validation_seq.npz"


def _tiny(tmp_path: Path, architecture: str = "lstm", T: int = 30):
    hp = {"architecture": architecture, "latent_dim": 104, "action_dim": 8, "sequence_length": 16}
    hp |= {"hidden_dim": 8} if architecture == "lstm" else {"d_model": 8, "nhead": 2, "num_layers": 1,
                                                             "dim_feedforward": 16, "dropout": 0.0}
    model_dir = tmp_path / f"wm_{architecture}"
    model_dir.mkdir()
    torch.manual_seed(0)
    torch.save(build_temporal_model(hp).state_dict(), model_dir / "world_model_best.pt")
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


def _dream_vs_rollout(model_dir, episodes, alignment):
    """(dream clipped returns, rollout_episode returns) over every window both define."""
    dream = CorridorDreamEnvironment(model_dir, seed_episodes_path=episodes, max_dream_steps=HORIZON,
                                     window_alignment=alignment)
    model, hp, scaler = load_temporal_model(model_dir)
    got = imagined_vs_real(dream)
    dream_r, ref_r = [], []
    for ep_id, ep in load_episodes(episodes).items():
        ref = rollout_episode_returns(model, scaler, ep, hp["sequence_length"], HORIZON)
        dream_r += list(got[ep_id][: len(ref), 0])
        ref_r += list(ref)
    return np.array(dream_r), np.array(ref_r)


# --------------------------------------------------------------------------- a) aligned == rollout_episode
@pytest.mark.parametrize("architecture", ["lstm", "transformer"])
def test_aligned_dream_matches_rollout_episode(tmp_path, architecture):
    dream_r, ref_r = _dream_vs_rollout(*_tiny(tmp_path, architecture), "aligned")
    assert len(ref_r) == 2 * (30 - 16 - 7 + 1)
    np.testing.assert_allclose(dream_r, ref_r, atol=1e-2, rtol=0)


@pytest.mark.parametrize("architecture", ["lstm", "transformer"])
def test_aligned_dream_matches_rollout_episode_with_real_models(tmp_path, architecture):
    model_dir = world_model_dir(architecture, 0)
    if not (model_dir / "world_model_best.pt").exists() or not VALIDATION_SEQ.exists():
        pytest.skip("v2 world models / sequence data not present (not committed)")
    with np.load(VALIDATION_SEQ) as d:
        mask = d["episode_id"] == 21000
        one = tmp_path / "one.npz"
        np.savez(one, **{k: d[k][mask] for k in d.files})
    dream_r, ref_r = _dream_vs_rollout(model_dir, one, "aligned")
    np.testing.assert_allclose(dream_r, ref_r, atol=1e-2, rtol=0)
    legacy_r, _ = _dream_vs_rollout(model_dir, one, "legacy")
    assert np.abs(legacy_r - ref_r).max() > 1.0  # the defect is there with the default


# --------------------------------------------------------------------------- b) legacy == published
@pytest.mark.parametrize("model_name", ["lstm_s0", "transformer_s0"])
def test_legacy_dream_reproduces_the_published_fidelity(model_name):
    arch, seed = model_name.split("_s")
    model_dir = world_model_dir(arch, int(seed))
    if not (model_dir / "world_model_best.pt").exists() or not VALIDATION_SEQ.exists():
        pytest.skip("v2 world models / sequence data not present (not committed)")
    published = json.loads(PUBLISHED_FIDELITY.read_text(encoding="utf-8"))["models"][model_name]
    for dream in (CorridorDreamEnvironment(model_dir, seed_episodes_path=VALIDATION_SEQ, max_dream_steps=HORIZON),
                  CorridorDreamEnvironment(model_dir, seed_episodes_path=VALIDATION_SEQ, max_dream_steps=HORIZON,
                                           window_alignment="legacy")):
        rows = imagined_vs_real(dream)
        pooled = np.concatenate([rows[e] for e in sorted(rows)])
        for col, label in ((0, "clipped"), (1, "raw")):
            assert np.corrcoef(pooled[:, col], pooled[:, 2])[0, 1] == pytest.approx(published[label]["pearson"], abs=1e-9)
            assert (pooled[:, col] - pooled[:, 2]).mean() == pytest.approx(published[label]["bias"], abs=1e-9)


# --------------------------------------------------------------------------- c) same interface, no SUMO
def test_both_alignments_share_the_interface_and_never_call_sumo(tmp_path, monkeypatch):
    import sumo_rl
    import traci

    from environments import corridor_environment

    def forbidden(*args, **kwargs):
        raise AssertionError("the Dream Environment called SUMO")

    for target, name in ((traci, "start"), (traci, "connect"), (traci, "init"), (traci, "getConnection"),
                         (sumo_rl.SumoEnvironment, "__init__"),
                         (corridor_environment.CorridorTrafficEnvironment, "__init__")):
        monkeypatch.setattr(target, name, forbidden)
    model_dir, episodes = _tiny(tmp_path)
    envs = {a: CorridorDreamEnvironment(model_dir, seed_episodes_path=episodes, eval_seed=0, window_alignment=a)
            for a in ("legacy", "aligned")}
    keys = {}
    for alignment, env in envs.items():
        assert env.action_space == envs["legacy"].action_space
        assert env.observation_space == envs["legacy"].observation_space
        obs, info = env.reset(options={"episode_id": 7, "start": 2})
        rng = np.random.default_rng(1)
        step_keys, done, steps = set(), False, 0
        while not done:
            obs, reward, terminated, truncated, step_info = env.step(rng.integers(0, 2, size=4))
            step_keys |= set(step_info)
            done, steps = truncated, steps + 1
        keys[alignment] = (set(info), step_keys, steps)
    assert keys["legacy"] == keys["aligned"]
    with pytest.raises(ValueError):
        CorridorDreamEnvironment(model_dir, seed_episodes_path=episodes, window_alignment="other")


# --------------------------------------------------------------------------- v1, checked only
V1_CHECKPOINT = ROOT_DIR / "models" / "checkpoints" / "world_model_best.pt"
V1_VALIDATION = ROOT_DIR / "datasets" / "processed" / "validation_latent.npz"


def _v1_returns():
    """v1 DreamEnvironment seeded with recorded windows and stepped with the recorded actions, next to
    rollout_episode and to a re-computation of the legacy shift. No v1 file is changed: the dream's
    window is set from outside, the way its own reset() would set it."""
    from environments import dream_environment as v1
    from evaluation.world_model_evaluation import load_reward_scaler, load_world_model, predict_next_step

    dream = v1.DreamEnvironment(latent_episodes_path=V1_VALIDATION)
    model, hp = load_world_model(V1_CHECKPOINT, torch.device("cpu"))
    scaler = load_reward_scaler(V1_CHECKPOINT)
    L, a_dim, clip = hp["sequence_length"], hp["action_dim"], (v1.REWARD_CLIP_MIN, v1.REWARD_CLIP_MAX)
    dream_r, ref_r, legacy_r = [], [], []
    for ep in list(load_episodes(V1_VALIDATION).values())[:3]:
        ref = rollout_episode_returns(model, scaler, ep, L, HORIZON, clip=clip)
        onehot = F.one_hot(torch.from_numpy(ep["actions"]), a_dim).float()
        for start in range(len(ref)):
            dream.reset()
            dream._window_z = torch.from_numpy(ep["z"][start:start + L]).clone()
            dream._window_actions_onehot = onehot[start:start + L].clone()
            total = 0.0
            for k in range(HORIZON):
                total += dream.step(int(ep["actions"][start + L - 1 + k]))[1]
            # The legacy shift, re-computed: the stored window moves on BEFORE the action is inserted.
            wz, stored, legacy = torch.from_numpy(ep["z"][start:start + L]), onehot[start:start + L].clone(), 0.0
            for k in range(HORIZON):
                a_k = onehot[start + L - 1 + k]
                aw = stored.clone()
                aw[-1] = a_k
                pz, pr = predict_next_step(model, wz, aw, torch.device("cpu"), scaler["reward_mean"], scaler["reward_std"])
                legacy += float(np.clip(pr.item(), *clip))
                stored = torch.cat([stored[1:], a_k[None]])
                wz = torch.cat([wz[1:], pz[None]])
            dream_r.append(total)
            legacy_r.append(legacy)
        ref_r += list(ref)
    return np.array(dream_r), np.array(ref_r), np.array(legacy_r)


@pytest.mark.skipif(not V1_CHECKPOINT.exists() or not V1_VALIDATION.exists(), reason="v1 weights / latent data not present")
def test_v1_dream_has_the_same_legacy_shift():
    dream_r, ref_r, legacy_r = _v1_returns()
    np.testing.assert_allclose(dream_r, legacy_r, atol=1e-3, rtol=0)   # it IS the legacy shift
    assert (np.abs(dream_r - ref_r) > 1e-2).sum() > 0                  # and differs from Experiment 1


@pytest.mark.skipif(not V1_CHECKPOINT.exists() or not V1_VALIDATION.exists(), reason="v1 weights / latent data not present")
@pytest.mark.xfail(strict=True, reason="documented defect: v1's DreamEnvironment shifts the action window before "
                                       "inserting the chosen action (ADDENDUM_PLANIFICACION.md, section 11); "
                                       "v1 files are deliberately not changed")
def test_v1_dream_matches_rollout_episode():
    dream_r, ref_r, _ = _v1_returns()
    np.testing.assert_allclose(dream_r, ref_r, atol=1e-2, rtol=0)
