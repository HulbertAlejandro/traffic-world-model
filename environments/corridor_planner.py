"""v2.1-A: decision-time planning with a trained world model, always from the REAL state
(docs/v2/ADDENDUM_PLANIFICACION.md, sections 1-4).

At every real control step the planner takes the real window of the last 16 normalized states and
the actions actually applied, scores the 16 joint actions in one batch -- each as the action of the
current state, followed by H - 1 imagined steps of a fixed continuation policy -- and returns the
one with the highest imagined return (sum of the reward the dream clips to [-345.43, 0.0]). Ties go
to the action with fewer switches ("keep" wins), then to the first in itertools.product order.

Windowing is Experiment 1's (evaluation/world_model_evaluation.py, rollout_episode): every state stays
paired with the action applied at it; after each imagined step the window shifts, the predicted state
enters last and the continuation's action takes the new last slot. It is NOT the legacy dream's
(CorridorDreamEnvironment with window_alignment="legacy" shifts the window before inserting the action
just taken, so from the third imagined step each state is paired with the previous step's action;
docs/v2/ADDENDUM_PLANIFICACION.md, section 11). tests/test_v2_planning.py checks that, fed the
recorded actions, the imagined returns match rollout_episode. Shared with the dream, imported: the
model loader, the reward clip and encode_actions. The planner never touches SUMO: the caller applies
its action.

Episode start (fewer than 16 real states): the window is padded at the front with copies of the
initial state paired with "keep" (section 4).
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from datasets.latent_sequence_dataset import encode_actions
from environments.corridor_dream_environment import REWARD_CLIP_MAX, REWARD_CLIP_MIN, load_temporal_model

N_SIGNALS = 4
JOINT_ACTIONS = np.array(list(itertools.product((0, 1), repeat=N_SIGNALS)), dtype=np.int64)  # (16, 4), A0 first
KEEP = np.zeros(N_SIGNALS, dtype=np.int64)

# continuation(step_index, predicted_states (N, 104) tensor) -> joint actions (N, 4) int64
Continuation = Callable[[int, torch.Tensor], np.ndarray]


def keep_continuation(step: int, predicted: torch.Tensor) -> np.ndarray:
    """plan_solo: keep the current phase on the four signals."""
    return np.zeros((len(predicted), N_SIGNALS), dtype=np.int64)


def ppo_continuation(ppo_model) -> Continuation:
    """plan_ppo: a Phase 3 dream PPO, deterministic, on the imagined (normalized) state."""
    def continuation(step: int, predicted: torch.Tensor) -> np.ndarray:
        actions, _ = ppo_model.predict(predicted.numpy(), deterministic=True)
        return np.asarray(actions, dtype=np.int64).reshape(len(predicted), N_SIGNALS)
    return continuation


def choose(returns: np.ndarray, actions: np.ndarray = JOINT_ACTIONS) -> int:
    """Index of the best action: highest return; ties -> fewest switches -> first in order."""
    returns = np.asarray(returns)
    tied = np.flatnonzero(returns == returns.max())
    return int(tied[np.argmin(actions[tied].sum(axis=1))])


@torch.no_grad()
def imagined_returns(model: torch.nn.Module, reward_mean: float, reward_std: float, window_z: torch.Tensor,
                     window_actions: torch.Tensor, first_actions: np.ndarray, continuation: Continuation,
                     horizon: int) -> np.ndarray:
    """Imagined H-step return of each first action, in one batch.

    ``window_z`` (L, D) and ``window_actions`` (L, action_dim) are the real window; the last action
    row belongs to the current state and is replaced by each first action.
    """
    n, action_dim = len(first_actions), window_actions.shape[1]
    z = window_z.unsqueeze(0).repeat(n, 1, 1)
    a = window_actions.unsqueeze(0).repeat(n, 1, 1).clone()
    a[:, -1] = encode_actions(torch.as_tensor(np.asarray(first_actions, dtype=np.int64)), action_dim)
    total = torch.zeros(n, dtype=torch.float64)
    for h in range(horizon):
        pred_z, pred_r = model(z, a)
        reward = pred_r * reward_std + reward_mean
        total += torch.clamp(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX).double()
        if h == horizon - 1:
            break
        nxt = encode_actions(torch.as_tensor(continuation(h, pred_z)), action_dim)
        z = torch.cat([z[:, 1:], pred_z.unsqueeze(1)], dim=1)
        a = torch.cat([a[:, 1:], nxt.unsqueeze(1)], dim=1)
    return total.numpy()


class CorridorPlanner:
    """Plans every real decision with one world model. ``act`` takes the current normalized state."""

    def __init__(self, model_dir: str | Path, horizon: int, continuation: Continuation = keep_continuation) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        self.model, hparams, scaler = load_temporal_model(model_dir)
        self.architecture = hparams["architecture"]
        self.sequence_length = hparams["sequence_length"]
        self.action_dim = hparams["action_dim"]
        self.reward_mean, self.reward_std = scaler["reward_mean"], scaler["reward_std"]
        self.horizon = horizon
        self.continuation = continuation
        self.reset()

    def reset(self) -> None:
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self.last_returns: np.ndarray | None = None
        self.decision_seconds: list[float] = []

    def window(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The current real window (L states, L actions; the last action row is a placeholder),
        padded at the front with the initial state and "keep" while there are fewer than L states."""
        L = self.sequence_length
        # Actions applied at the states BEFORE the current one (also right after act() appended its own).
        applied = self._actions[:len(self._states) - 1]
        states, actions = self._states[-L:], applied[-(L - 1):] if L > 1 else []
        pad = L - len(states)
        states = [self._states[0]] * pad + states
        actions = [KEEP] * pad + list(actions) + [KEEP]  # placeholder for the current state's action
        window_z = torch.as_tensor(np.asarray(states, dtype=np.float32))
        window_a = encode_actions(torch.as_tensor(np.asarray(actions, dtype=np.int64)), self.action_dim)
        return window_z, window_a

    def act(self, state: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self._states.append(np.asarray(state, dtype=np.float32))
        window_z, window_a = self.window()
        returns = imagined_returns(self.model, self.reward_mean, self.reward_std, window_z, window_a,
                                   JOINT_ACTIONS, self.continuation, self.horizon)
        action = JOINT_ACTIONS[choose(returns)].copy()
        self._actions.append(action)
        self.last_returns = returns
        self.decision_seconds.append(time.perf_counter() - t0)
        return action
