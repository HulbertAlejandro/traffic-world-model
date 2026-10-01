"""Real SUMO interactions of every new seed (ADDENDUM.md), from the saved artifacts:
training steps = num_timesteps of the final model (real for direct RL, imagined for the Dream PPO),
selection/evaluation steps = sum of ep_lengths in evaluations.npz (always real SUMO).
Expected: direct 10k 10,240 + 3,000 = 13,240; direct 30k 30,208 + 9,000 = 39,208; Dream 0 real training
steps + 3,000 selection steps (its 50,000+ training steps are imagined)."""
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[3]
CK = ROOT / "models" / "checkpoints"
SPEC = {"controller_10seeds": (range(3, 10), "ppo_controller_final.zip", 0, 3000),
        "controller_direct_10seeds": (range(4, 10), "ppo_controller_direct_final.zip", 10240, 3000),
        "controller_direct_30k_10seeds": (range(4, 10), "ppo_controller_direct_final.zip", 30208, 9000)}


def main():
    ok, out = True, {}
    for folder, (seeds, final, exp_train, exp_eval) in SPEC.items():
        for s in seeds:
            d = CK / folder / f"seed{s}"
            steps = PPO.load(d / final, device="cpu").num_timesteps
            with np.load(d / "evaluations.npz") as e:
                ev = int(e["ep_lengths"].sum())
            hp = json.loads((d / "best_model.json").read_text(encoding="utf-8"))
            real_train = steps if "direct" in folder else 0
            good = (real_train == exp_train) and ev == exp_eval and hp["seed"] == s
            ok &= good
            out[f"{folder}/seed{s}"] = {"train_steps": steps, "real_train_steps": real_train, "eval_steps": ev,
                                        "real_total": real_train + ev, "json_seed": hp["seed"]}
            print(f"{folder}/seed{s}: entrenamiento {steps} pasos ({'reales' if real_train else 'imaginados'}), "
                  f"evaluacion {ev} pasos reales, total real {real_train + ev}, seed en .json {hp['seed']} "
                  f"{'OK' if good else 'DIFERENTE'}")
    (Path(__file__).parent / "interactions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("interacciones OK" if ok else "interacciones DIFERENTES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
