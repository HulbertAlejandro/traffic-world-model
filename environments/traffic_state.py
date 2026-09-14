"""Domain-level traffic state abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class TrafficState:
    """Structured state for a single-intersection traffic signal controller.

    The object represents the domain-level view of the intersection state, while
    ``to_vector()`` converts it to a flat numpy array suitable for model input.

    The signal phase is represented as a one-hot vector (``phase_one_hot``)
    rather than a raw scalar index. Phases are categories without a numeric
    order (phase 2 is not "more" than phase 1), so a scalar encoding would
    falsely imply an ordinal relationship to any model trained on this vector.
    """

    vehicle_counts: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    queue_lengths: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    waiting_times: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    mean_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    occupancies: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    phase_one_hot: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    elapsed_phase_time: float = 0.0
    remaining_phase_time: float = 0.0

    def __post_init__(self) -> None:
        """Ensure all per-lane arrays are float32 and share the same lane count."""
        for name in (
            "vehicle_counts",
            "queue_lengths",
            "waiting_times",
            "mean_speeds",
            "occupancies",
        ):
            value = getattr(self, name)
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")

            expected_size = len(self.vehicle_counts)
            if arr.shape[0] != expected_size:
                raise ValueError(
                    f"{name} length mismatch: expected {expected_size}, got {arr.shape[0]}"
                )
            setattr(self, name, arr)

        phase_arr = np.asarray(self.phase_one_hot, dtype=np.float32)
        if phase_arr.ndim != 1:
            raise ValueError(f"phase_one_hot must be a 1D array, got shape {phase_arr.shape}")
        self.phase_one_hot = phase_arr

    def to_vector(self) -> np.ndarray:
        """Flatten the structured state into a numeric vector."""
        return np.concatenate(
            [
                self.vehicle_counts,
                self.queue_lengths,
                self.waiting_times,
                self.mean_speeds,
                self.occupancies,
                self.phase_one_hot,
                np.array(
                    [float(self.elapsed_phase_time), float(self.remaining_phase_time)],
                    dtype=np.float32,
                ),
            ]
        )

    @property
    def size(self) -> int:
        """Return the vectorized state length."""
        return len(self.vehicle_counts) * 5 + len(self.phase_one_hot) + 2