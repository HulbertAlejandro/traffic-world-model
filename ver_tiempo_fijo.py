from environments.encoded_traffic_environment import EncodedTrafficEnvironment
from configs.environment import EnvironmentConfig
from scripts.evaluate_multiseed_statistical import build_action_functions, parse_policy
import traci

# Same environment, scenario and window as ver_controlador.py; only the decision changes.
# The fixed-time policy is the official baseline of the published tables, imported from
# scripts/evaluate_multiseed_statistical.py (action 1 every 5th step, 0 otherwise). No PPO is loaded.
env = EncodedTrafficEnvironment(environment_config=EnvironmentConfig(use_gui=True))
(_, fixed_time), = build_action_functions(parse_policy('tiempo_fijo=fixed'), env)

obs, info = env.reset(seed=7025)

(xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
traci.gui.setBoundary('View #0', xmin, ymin, xmax, ymax)

input('Vista centrada automaticamente. Ahora sube el valor de Delay en la barra superior de la ventana de SUMO (por ejemplo 200), y presiona Enter aqui para empezar...')

terminated = truncated = False
step = 0
while not (terminated or truncated):
    action = fixed_time(None, step)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

env.close()
