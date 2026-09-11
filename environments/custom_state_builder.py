"""
Custom traffic state representation.

Este módulo construye el vector de estado del proyecto usando
información obtenida directamente desde TraCI.

NO utiliza la observación de sumo-rl.
"""

from __future__ import annotations

import numpy as np
import traci

from environments.traffic_state import TrafficState


class CustomStateBuilder:
    """
    Construye el estado del proyecto a partir de TraCI.

    El estado se modela como ``TrafficState`` y luego se serializa a un vector
    ``numpy.ndarray`` para compatibilidad con el resto del pipeline.
    """

    def __init__(self, traffic_light_id: str | None = None) -> None:
        self.traffic_light_id = traffic_light_id
        self._lane_ids: list[str] | None = None

    def _resolve_traffic_light_id(self) -> str:
        """Resolve the traffic signal ID only when the simulation is connected."""
        traffic_light_id = self.traffic_light_id
        if traffic_light_id is not None:
            return traffic_light_id

        signal_ids = traci.trafficlight.getIDList()
        if not signal_ids:
            raise RuntimeError("No traffic lights are available in the SUMO simulation.")
        self.traffic_light_id = signal_ids[0]
        return self.traffic_light_id

    def build(
        self,
        observation=None,
        env=None,
    ) -> np.ndarray:
        """Build the project state from TraCI and return the flat vector."""
        _ = observation
        _ = env
        return self.build_state().to_vector()

    def state_size(self) -> int:
        """Return the flattened vector dimension for the current set of lanes."""
        if self._lane_ids is None:
            return 0
        return len(self._lane_ids) * 5 + 2

    def build_state(self) -> TrafficState:
        """Crear el objeto de dominio ``TrafficState`` desde TraCI."""
        traffic_light_id = self._resolve_traffic_light_id()
        if self._lane_ids is None:
            self._lane_ids = self._incoming_lanes(traffic_light_id)

        lane_ids = self._lane_ids
        vehicle_count_size = len(lane_ids)

        vehicle_counts = np.empty(vehicle_count_size, dtype=np.float32)
        queue_lengths = np.empty(vehicle_count_size, dtype=np.float32)
        waiting_times = np.empty(vehicle_count_size, dtype=np.float32)
        mean_speeds = np.empty(vehicle_count_size, dtype=np.float32)
        occupancies = np.empty(vehicle_count_size, dtype=np.float32)

        for i, lane in enumerate(lane_ids):
            vehicle_counts[i] = traci.lane.getLastStepVehicleNumber(lane)
            queue_lengths[i] = traci.lane.getLastStepHaltingNumber(lane)
            waiting_times[i] = traci.lane.getWaitingTime(lane)
            mean_speeds[i] = traci.lane.getLastStepMeanSpeed(lane)
            occupancies[i] = traci.lane.getLastStepOccupancy(lane)

        current_phase = float(traci.trafficlight.getPhase(traffic_light_id))
        remaining_phase_time = float(
            traci.trafficlight.getNextSwitch(traffic_light_id)
            - traci.simulation.getTime()
        )

        return TrafficState(
            vehicle_counts=vehicle_counts,
            queue_lengths=queue_lengths,
            waiting_times=waiting_times,
            mean_speeds=mean_speeds,
            occupancies=occupancies,
            current_phase=current_phase,
            remaining_phase_time=remaining_phase_time,
        )

    def _incoming_lanes(self, traffic_light_id: str | None = None) -> list[str]:
        """Devuelve únicamente los carriles de entrada al semáforo."""

        if traffic_light_id is None:
            traffic_light_id = self._resolve_traffic_light_id()

        controlled_links = traci.trafficlight.getControlledLinks(traffic_light_id)

        lanes = []

        for links in controlled_links:
            if not links:
                continue

            incoming_lane = links[0][0]

            if incoming_lane not in lanes:
                lanes.append(incoming_lane)

        lanes.sort()

        return lanes