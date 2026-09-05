"""
Ewaluacja deterministyczna wytrenowanej polityki lokomocji (PPO).
Skrypt umozliwia interaktywna walidacje jakosci sterowania holonomicznego
robota szescionoznego w srodowisku graficznym silnika PyBullet (GUI).
"""

import time
from pathlib import Path
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from hexapod_learning import HexapodRLTrainingEnv

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hexapod.zip"
POLICY_FREQUENCY = 50.0  # Hz
CYCLE_TIME = 1.0 / POLICY_FREQUENCY
DEADZONE_VELOCITY = 0.01  # m/s

def main():
    print(f"[INFO] Ladowanie wytrenowanego modelu PPO z: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie odnaleziono pliku wag sieci: {MODEL_PATH}. "
            "Upewnij sie, ze model znajduje sie w odpowiednim katalogu."
        )

    model = PPO.load(str(MODEL_PATH), device="cpu")

    print("[INFO] Inicjalizacja srodowiska symulacyjnego w trybie GUI...")
    env = HexapodRLTrainingEnv(render_mode="human", enable_noise=False)
    obs, info = env.reset()

    slider_vy = p.addUserDebugParameter("Vy (Przod)", -0.3, 0.3, 0.0)
    slider_vx = p.addUserDebugParameter("Vx (Bok)", -0.3, 0.3, 0.0)

    zero_action = np.zeros(18, dtype=np.float32)

    print("[INFO] Rozpoczeto petle decyzyjna inferencji czasu rzeczywistego (50 Hz).")
    print("[INFO] Zmiana wektora predkosci referencyjnej sterowana z poziomu suwakow GUI.")

    try:
        while True:
            cmd_vy = p.readUserDebugParameter(slider_vy)
            cmd_vx = p.readUserDebugParameter(slider_vx)

            env.unwrapped.target_cmd_vy = cmd_vy
            env.unwrapped.target_cmd_vx = cmd_vx

            env.unwrapped.steps_until_target_change = int(1e6)
            env.unwrapped.step_counter = 0  

            if abs(cmd_vx) < DEADZONE_VELOCITY and abs(cmd_vy) < DEADZONE_VELOCITY:
                action = zero_action
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            time.sleep(CYCLE_TIME)

            if terminated:
                print("[OSTRZEZENIE] Naruszenie warunku stabilnosci postawy. Reinicjalizacja stanu bazowego.")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n[INFO] Przerwano wykonywanie procedury przez uzytkownika.")
    finally:
        print("[INFO] Zamykanie instancji serwera silnika fizycznego.")
        env.close()

if __name__ == "__main__":
    main()