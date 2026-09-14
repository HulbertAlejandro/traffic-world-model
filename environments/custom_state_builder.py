"""
Custom traffic state representation.

Este módulo construye el vector de estado del proyecto usando información
obtenida directamente desde TraCI. NO utiliza la observación de sumo-rl.
"""

from __future__ import annotations

import numpy as np
import traci

from environments.traffic_state import TrafficState


class CustomStateBuilder:
    """Construye el estado del proyecto a partir de TraCI.

    El estado se modela como ``TrafficState`` y luego se serializa a un vector
    ``numpy.ndarray`` para compatibilidad con el resto del pipeline.
    """

    def __init__(self, traffic_light_id: str | None = None) -> None:
        self.traffic_light_id = traffic_light_id
        self._lane_ids: list[str] | None = None
        self._num_phases: int | None = None

    def _resolve_traffic_light_id(self) -> str:
        """Resolve the traffic signal ID only when the simulation is connected."""
        if self.traffic_light_id is not None:
            return self.traffic_light_id

        signal_ids = traci.trafficlight.getIDList()
        if not signal_ids:
            raise RuntimeError("No traffic lights are available in the SUMO simulation.")
        self.traffic_light_id = signal_ids[0]
        return self.traffic_light_id

    def _resolve_num_phases(self, traffic_light_id: str) -> int:
        """Read the number of phases from the currently loaded program logic."""
        logics = traci.trafficlight.getAllProgramLogics(traffic_light_id)
        if not logics:
            raise RuntimeError(f"No program logic found for traffic light '{traffic_light_id}'.")
        return len(logics[0].getPhases())

    def build(self, observation=None, env=None) -> np.ndarray:
        """Build the project state from TraCI and return the flat vector.

        ``observation`` and ``env`` are accepted (and ignored) so this class
        stays a drop-in replacement for ``DefaultStateBuilder.build(observation, env)``.
        """
        _ = observation
        _ = env

        state = self.build_state()
        vector = state.to_vector()

        expected_size = self.state_size()
        if expected_size and vector.size != expected_size:
            raise RuntimeError(
                f"State vector size mismatch: expected {expected_size}, got {vector.size}. "
                "This usually means the number of lanes or phases changed mid-run."
            )
        return vector

    def state_size(self) -> int:
        """Return the flattened vector dimension for the current lanes/phases."""
        if self._lane_ids is None or self._num_phases is None:
            return 0
        return len(self._lane_ids) * 5 + self._num_phases + 2

    def build_state(self) -> TrafficState:
        """Crear el objeto de dominio ``TrafficState`` desde TraCI."""
        traffic_light_id = self._resolve_traffic_light_id()
        if self._lane_ids is None:
            self._lane_ids = self._incoming_lanes(traffic_light_id)
        if self._num_phases is None:
            self._num_phases = self._resolve_num_phases(traffic_light_id)

        lane_ids = self._lane_ids
        n = len(lane_ids)

        vehicle_counts = np.empty(n, dtype=np.float32)
        queue_lengths = np.empty(n, dtype=np.float32)
        waiting_times = np.empty(n, dtype=np.float32)
        mean_speeds = np.empty(n, dtype=np.float32)
        occupancies = np.empty(n, dtype=np.float32)

        for i, lane in enumerate(lane_ids):
            vehicle_counts[i] = traci.lane.getLastStepVehicleNumber(lane)
            queue_lengths[i] = traci.lane.getLastStepHaltingNumber(lane)
            waiting_times[i] = traci.lane.getWaitingTime(lane)
            mean_speeds[i] = traci.lane.getLastStepMeanSpeed(lane)
            occupancies[i] = traci.lane.getLastStepOccupancy(lane)

        current_phase_index = traci.trafficlight.getPhase(traffic_light_id)
        phase_one_hot = np.zeros(self._num_phases, dtype=np.float32)
        if 0 <= current_phase_index < self._num_phases:
            phase_one_hot[current_phase_index] = 1.0

        remaining_phase_time = max(
            0.0,
            float(traci.trafficlight.getNextSwitch(traffic_light_id) - traci.simulation.getTime()),
        )
        phase_duration = float(traci.trafficlight.getPhaseDuration(traffic_light_id))
        elapsed_phase_time = max(0.0, phase_duration - remaining_phase_time)

        return TrafficState(
            vehicle_counts=vehicle_counts,
            queue_lengths=queue_lengths,
            waiting_times=waiting_times,
            mean_speeds=mean_speeds,
            occupancies=occupancies,
            phase_one_hot=phase_one_hot,
            elapsed_phase_time=elapsed_phase_time,
            remaining_phase_time=remaining_phase_time,
        )

    def _incoming_lanes(self, traffic_light_id: str | None = None) -> list[str]:
        """Devuelve únicamente los carriles de entrada al semáforo, sin duplicados."""
        if traffic_light_id is None:
            traffic_light_id = self._resolve_traffic_light_id()

        controlled_links = traci.trafficlight.getControlledLinks(traffic_light_id)
        lanes: list[str] = []
        for links in controlled_links:
            if not links:
                continue
            incoming_lane = links[0][0]
            if incoming_lane not in lanes:
                lanes.append(incoming_lane)
        lanes.sort()
        return lanes