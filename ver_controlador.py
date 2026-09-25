from stable_baselines3 import PPO
from environments.encoded_traffic_environment import EncodedTrafficEnvironment
from configs.environment import EnvironmentConfig
import traci

env = EncodedTrafficEnvironment(environment_config=EnvironmentConfig(use_gui=True))
model = PPO.load('models/checkpoints/controller/best_model.zip')

obs, info = env.reset(seed=3000)

(xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
traci.gui.setBoundary('View #0', xmin, ymin, xmax, ymax)

input('Vista centrada automaticamente. Ahora sube el valor de Delay en la barra superior de la ventana de SUMO (por ejemplo 200), y presiona Enter aqui para empezar...')

terminated = truncated = False
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))

env.close()
