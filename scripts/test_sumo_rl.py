"""
Script de verificación: confirma que sumo-rl puede controlar NUESTRA
propia red (single-intersection construida con netgenerate), conectando
via TraCI, con el entorno tipo Gymnasium respondiendo a step/reset.

Este script NO es parte del pipeline final del proyecto (no entrena nada).
Es un smoke test para validar que la red propia funciona igual de bien
que la red de ejemplo que usamos para la verificación inicial.
"""

import time
import sumo_rl

# Nuestra propia red (generada con netgenerate + rou.xml corregido a 350 veh/h/brazo)
NET_FILE = "environments/single-intersection/single-intersection.net.xml"
ROUTE_FILE = "environments/single-intersection/single-intersection.rou.xml"

env = sumo_rl.SumoEnvironment(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    use_gui=True,           # activado: se abrirá la ventana de sumo-gui
    num_seconds=300,        # 5 minutos simulados, suficiente para observar varios ciclos
    single_agent=True,
)

obs, info = env.reset()
print("Observación inicial:")
print(obs)
print("Shape de la observación:", obs.shape)

done = False
step_count = 0
total_reward = 0.0

while not done and step_count < 100:
    action = env.action_space.sample()  # acción aleatoria, solo para probar
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    step_count += 1
    print(f"Paso {step_count} | acción={action} | recompensa={reward:.3f}")
    time.sleep(0.3)  # pausa para poder observar la simulación en la GUI

print("\n--- Verificación completada ---")
print(f"Pasos ejecutados: {step_count}")
print(f"Recompensa acumulada: {total_reward:.3f}")

env.close()