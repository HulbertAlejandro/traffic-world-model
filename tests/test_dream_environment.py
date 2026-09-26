from __future__ import annotations

import numpy as np
import pytest
import torch

from environments.dream_environment import DreamEnvironment
from models.world_model import LatentDynamicsLSTM


LATENT_DIM = 4
ACTION_DIM = 2
SEQUENCE_LENGTH = 5


def _make_checkpoint(tmp_path):
    model = LatentDynamicsLSTM(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM, sequence_length=SEQUENCE_LENGTH
    )
    checkpoint_path = tmp_path / "world_model_best.pt"
    torch.save(model.state_dict(), checkpoint_path)
    hparams_path = checkpoint_path.with_suffix(".json")
    hparams_path.write_text(
        f'{{"latent_dim": {LATENT_DIM}, "action_dim": {ACTION_DIM}, '
        f'"hidden_dim": 128, "sequence_length": {SEQUENCE_LENGTH}}}',
        encoding="utf-8",
    )
    (checkpoint_path.parent / "reward_scaler.json").write_text(
        '{"reward_mean": 0.0, "reward_std": 1.0}', encoding="utf-8"
    )
    return checkpoint_path


def _make_latent_episodes(tmp_path, episode_lengths):
    rng = np.random.default_rng(0)
    z, actions, rewards, episode_id, time_step = [], [], [], [], []
    for ep_id, length in enumerate(episode_lengths):
        z.append(rng.normal(size=(length, LATENT_DIM)).astype(np.float32))
        actions.append(rng.integers(0, ACTION_DIM, size=length).astype(np.int64))
        rewards.append(rng.normal(size=length).astype(np.float32))
        episode_id.append(np.full(length, ep_id, dtype=np.int64))
        time_step.append(np.arange(length, dtype=np.int64))

    path = tmp_path / "train_latent.npz"
    np.savez(
        path,
        z=np.concatenate(z),
        next_z=np.concatenate(z),  # unused by load_episodes, placeholder
        actions=np.concatenate(actions),
        rewards=np.concatenate(rewards),
        episode_id=np.concatenate(episode_id),
        time_step=np.concatenate(time_step),
    )
    return path


def test_reset_returns_valid_observation(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[10])

    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=3)
    observation, info = env.reset(seed=0)

    assert observation.shape == (LATENT_DIM,)
    assert info["seeded_from_real_data"] is True


def test_eval_seed_makes_seed_windows_reproducible(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[10, 12, 15])

    def first_observations(eval_seed):
        env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=3, eval_seed=eval_seed)
        return [env.reset()[0] for _ in range(5)]

    same_a, same_b, other = first_observations(123), first_observations(123), first_observations(456)
    assert all(np.array_equal(a, b) for a, b in zip(same_a, same_b))
    assert not all(np.array_equal(a, c) for a, c in zip(same_a, other))


def test_step_returns_valid_transition(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[10])

    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=3)
    env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(1)

    assert observation.shape == (LATENT_DIM,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["dream_step"] == 1


def test_episode_truncates_at_max_dream_steps(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[10])

    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=3)
    env.reset(seed=0)

    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is False

    assert truncated is True


def test_rejects_episodes_shorter_than_sequence_length(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[2])  # shorter than SEQUENCE_LENGTH=5

    with pytest.raises(ValueError, match="No episode"):
        DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=3)


def test_rejects_invalid_action(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[10])

    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=3)
    env.reset(seed=0)

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)


def test_action_streak_tracked_in_info(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[10])

    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=3)
    env.reset(seed=0)

    _, _, _, _, info = env.step(1)
    assert info["consecutive_action_streak"] == 1
    _, _, _, _, info = env.step(1)
    assert info["consecutive_action_streak"] == 2
    _, _, _, _, info = env.step(0)
    assert info["consecutive_action_streak"] == 1


def test_imagined_reward_is_clipped_to_empirical_range(tmp_path, monkeypatch):
    import environments.dream_environment as dream_env_module

    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[10])

    monkeypatch.setattr(dream_env_module, "REWARD_CLIP_MIN", -1.0)
    monkeypatch.setattr(dream_env_module, "REWARD_CLIP_MAX", 1.0)

    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=3)
    env.reset(seed=0)

    for _ in range(3):
        _, reward, _, _, _ = env.step(1)
        assert -1.0 <= reward <= 1.0


def test_raw_predicted_reward_exposed_in_info(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[10])

    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=3)
    env.reset(seed=0)

    _, reward, _, _, info = env.step(1)
    assert "raw_predicted_reward" in info
    assert isinstance(info["raw_predicted_reward"], float)
    # el valor reportado como reward YA debe estar recortado; el crudo puede
    # coincidir o no dependiendo de si el recorte se activó
    if info["reward_clipped"]:
        assert info["raw_predicted_reward"] != reward
    else:
        assert abs(info["raw_predicted_reward"] - reward) < 1e-5


def test_dream_training_never_touches_sumo_or_traci(tmp_path, monkeypatch):
    """A full imagined episode and a short PPO training run inside the Dream
    Environment must not start, connect to, or step SUMO in any way."""
    import subprocess

    import sumo_rl
    import traci
    from stable_baselines3 import PPO

    calls = []

    def forbidden(name):
        def _raise(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} called during a Dream Environment episode")
        return _raise

    for name in ("start", "connect", "init", "switch", "getConnection", "simulationStep", "load"):
        monkeypatch.setattr(traci, name, forbidden(f"traci.{name}"))
    monkeypatch.setattr(sumo_rl.SumoEnvironment, "__init__", forbidden("sumo_rl.SumoEnvironment"))
    real_popen = subprocess.Popen

    class NoSumoPopen(real_popen):
        def __init__(self, args, *a, **k):
            if "sumo" in str(args).lower():
                forbidden(f"subprocess.Popen({args!r})")()
            super().__init__(args, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", NoSumoPopen)

    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[20, 20])
    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=5)

    env.reset(seed=0)
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(1)
        assert info["imagined"] is True

    PPO("MlpPolicy", env, n_steps=16, batch_size=8, n_epochs=1, seed=0, verbose=0).learn(total_timesteps=48)

    assert calls == []
