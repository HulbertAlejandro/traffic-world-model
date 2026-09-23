"""
Custom traffic state representation.

Este módulo construye el vector de estado del proyecto usando información
obtenida directamente desde TraCI. NO utiliza la observación de sumo-rl.

Excepción documentada: la fase actual del semáforo y el tiempo asociado a ella
NO se leen de traci.trafficlight.getPhase()/getNextSwitch()/getPhaseDuration().
sumo_rl controla el semáforo escribiendo el estado rojo/ámbar/verde directamente
como cadena de texto (TrafficSignal.set_next_phase -> setRedYellowGreenState),
sin usar el programa de fases nativo del .net.xml. Como consecuencia, esas
llamadas de TraCI nunca reflejan los cambios reales que produce el agente --
quedan congeladas en el estado inicial del programa cargado (se confirmó
empíricamente: getPhase() se mantuvo en 0 mientras el semáforo físico sí
cambiaba, y getNextSwitch() reportaba un valor fijo de 86400s). Por eso esta
clase lee la fase y el tiempo transcurrido directamente de los atributos que
sumo_rl sí mantiene actualizados (TrafficSignal.green_phase y
TrafficSignal.time_since_last_phase_change), obtenidos a través de la
referencia al entorno sumo_rl que TrafficEnvironment provee en build_state().
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

        ``observation`` is accepted (and ignored) to keep the
        ``build(observation, env)`` call signature.
        ``env`` (the sumo_rl environment) is forwarded to ``build_state`` --
        see that method's docstring for why it is needed.
        """
        _ = observation

        state = self.build_state(sumo_rl_env=env)
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

    def build_state(self, sumo_rl_env=None) -> TrafficState:
        """Crear el objeto de dominio ``TrafficState`` desde TraCI.

        Args:
            sumo_rl_env: la instancia de ``sumo_rl.SumoEnvironment`` en uso
                (``TrafficEnvironment.env``). Se usa ÚNICAMENTE para leer la
                fase real (``TrafficSignal.green_phase``) y el tiempo
                transcurrido desde el último cambio
                (``TrafficSignal.time_since_last_phase_change``), que son los
                valores que sumo_rl realmente mantiene actualizados al aplicar
                acciones. Si no se provee (por ejemplo, en un test que
                construye este builder de forma aislada), se cae de vuelta a
                TraCI puro para esos dos valores -- con la advertencia de que
                en ese caso no reflejarán acciones aplicadas por sumo_rl.
        """
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

        current_phase_index, elapsed_phase_time, remaining_phase_time = self._resolve_phase_timing(
            traffic_light_id, sumo_rl_env
        )

        phase_one_hot = np.zeros(self._num_phases, dtype=np.float32)
        if 0 <= current_phase_index < self._num_phases:
            phase_one_hot[current_phase_index] = 1.0

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

    def _resolve_phase_timing(
        self, traffic_light_id: str, sumo_rl_env
    ) -> tuple[int, float, float]:
        """Return (phase_index, elapsed_time, remaining_time), from sumo_rl when available."""
        traffic_signal = None
        if sumo_rl_env is not None:
            traffic_signal = getattr(sumo_rl_env, "traffic_signals", {}).get(traffic_light_id)

        if traffic_signal is not None:
            phase_index = int(traffic_signal.green_phase)
            elapsed = float(traffic_signal.time_since_last_phase_change)
            # No fixed cycle exists under RL control; "remaining" is redefined
            # as time left until a phase switch would even be legal (the real
            # constraint the controller operates under), not a scheduled
            # switch time (which sumo_rl does not use).
            remaining = max(0.0, float(traffic_signal.min_green) - elapsed)
            return phase_index, elapsed, remaining

        # Fallback for standalone use without a sumo_rl environment reference
        # (e.g. unit tests constructing this builder in isolation). Known to
        # NOT reflect sumo_rl-driven phase changes -- see module docstring.
        phase_index = traci.trafficlight.getPhase(traffic_light_id)
        remaining = max(
            0.0,
            float(traci.trafficlight.getNextSwitch(traffic_light_id) - traci.simulation.getTime()),
        )
        phase_duration = float(traci.trafficlight.getPhaseDuration(traffic_light_id))
        elapsed = max(0.0, phase_duration - remaining)
        return phase_index, elapsed, remaining

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