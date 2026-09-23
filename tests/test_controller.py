from __future__ import annotations

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from configs import ControllerConfig
from environments.dream_environment import DreamEnvironment
from models.world_model import LatentDynamicsLSTM
from training.train_controller import build_normalized_envs


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
        next_z=np.concatenate(z),
        actions=np.concatenate(actions),
        rewards=np.concatenate(rewards),
        episode_id=np.concatenate(episode_id),
        time_step=np.concatenate(time_step),
    )
    return path


def test_controller_config_validates_defaults():
    config = ControllerConfig()
    assert config.total_timesteps > 0
    assert 0.0 < config.gamma <= 1.0


def test_controller_config_rejects_invalid_gamma():
    with pytest.raises(ValueError, match="gamma"):
        ControllerConfig(gamma=1.5)


def test_controller_config_rejects_invalid_dream_max_steps():
    with pytest.raises(ValueError, match="dream_max_steps"):
        ControllerConfig(dream_max_steps=0)


def test_controller_config_rejects_invalid_reward_clip():
    with pytest.raises(ValueError, match="reward_clip"):
        ControllerConfig(reward_clip=0.0)


def test_normalized_env_stack_trains_and_syncs_through_eval_callback(tmp_path):
    """The exact wrapping used by both training scripts, with synthetic models
    (a DreamEnvironment stands in for the SUMO-based eval env). EvalCallback
    calls sync_envs_normalization before each evaluation and asserts both stacks
    are wrapped the same way, so a mis-wrapped eval env fails here, not after
    minutes of real training."""
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[20, 20])
    config = ControllerConfig(n_steps=16, batch_size=8, n_epochs=1)

    train_env, eval_env = build_normalized_envs(
        lambda: DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=5),
        lambda: DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=5, eval_seed=0),
        config,
    )
    model = PPO("MlpPolicy", train_env, n_steps=16, batch_size=8, n_epochs=1, seed=0, verbose=0)
    # log_path is required for EvalCallback to record evaluations_timesteps
    # (both training scripts pass one too).
    callback = EvalCallback(
        eval_env, eval_freq=16, n_eval_episodes=2, deterministic=True, verbose=0, log_path=str(tmp_path)
    )
    model.learn(total_timesteps=48, callback=callback)

    assert len(callback.evaluations_timesteps) >= 1          # at least one evaluation ran
    assert isinstance(model.get_vec_normalize_env(), VecNormalize)
    assert train_env.ret_rms.count > 1                       # training stats were updated
    # eval_env has training=False, so it never updates its own statistics: a
    # non-trivial count here can only come from sync_envs_normalization.
    assert eval_env.ret_rms.count > 1
    assert eval_env.norm_reward is False and eval_env.training is False


def test_ppo_trains_a_few_steps_inside_dream_environment(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[20, 20])

    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=5)
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=8, n_epochs=1, seed=0, verbose=0)
    model.learn(total_timesteps=32)

    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert int(action) in (0, 1)


def test_saved_ppo_policy_can_be_reloaded(tmp_path):
    checkpoint_path = _make_checkpoint(tmp_path)
    latent_path = _make_latent_episodes(tmp_path, episode_lengths=[20, 20])

    env = DreamEnvironment(checkpoint_path, latent_path, max_dream_steps=5)
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=8, n_epochs=1, seed=0, verbose=0)
    model.learn(total_timesteps=32)

    save_path = tmp_path / "ppo_test.zip"
    model.save(save_path)

    reloaded = PPO.load(save_path)
    obs, _ = env.reset(seed=0)
    action, _ = reloaded.predict(obs, deterministic=True)
    assert int(action) in (0, 1)
