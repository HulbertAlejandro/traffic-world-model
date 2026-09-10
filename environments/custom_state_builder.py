"""
Custom traffic state representation.

Este módulo construye el vector de estado del proyecto usando
información obtenida directamente desde TraCI.

NO utiliza la observación de sumo-rl.
"""

from __future__ import annotations

import numpy as np
import traci


class CustomStateBuilder:
    """
    Construye el estado del proyecto a partir de TraCI.

    Estado (17 variables):

        0-3   -> vehículos detenidos (cola) por aproximación
        4-7   -> tiempo de espera acumulado por aproximación
        8-11  -> velocidad media por aproximación
        12-15 -> ocupación por aproximación
        16    -> fase actual del semáforo
    """

    def __init__(self, traffic_light_id: str = "A0") -> None:
        self.traffic_light_id = traffic_light_id

    def build(self, observation, env) -> np.ndarray:
        """
        Construye el estado usando únicamente TraCI.
        """

        lane_ids = self._incoming_lanes()

        queue_lengths = []
        waiting_times = []
        mean_speeds = []
        occupancies = []

        for lane in lane_ids:

            queue_lengths.append(
                traci.lane.getLastStepHaltingNumber(lane)
            )

            waiting_times.append(
                traci.lane.getWaitingTime(lane)
            )

            mean_speeds.append(
                traci.lane.getLastStepMeanSpeed(lane)
            )

            occupancies.append(
                traci.lane.getLastStepOccupancy(lane)
            )

        current_phase = traci.trafficlight.getPhase(
            self.traffic_light_id
        )

        state = np.array(
            [
                *queue_lengths,
                *waiting_times,
                *mean_speeds,
                *occupancies,
                float(current_phase),
            ],
            dtype=np.float32,
        )

        return state

    def state_size(self) -> int:
        """
        Número de variables del estado.
        """
        return 17

    def _incoming_lanes(self) -> list[str]:
        """
        Devuelve únicamente los carriles de entrada al semáforo.
        """

        controlled_links = traci.trafficlight.getControlledLinks(
            self.traffic_light_id
        )

        lanes = []

        for links in controlled_links:

            if not links:
                continue

            incoming_lane = links[0][0]

            if incoming_lane not in lanes:
                lanes.append(incoming_lane)

        lanes.sort()

        return lanes