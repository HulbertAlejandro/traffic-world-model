"""Controls 1 and 2 of ADDENDUM.md: a re-trained run must reproduce an existing official seed exactly.

Compares (a) the EvalCallback log (evaluations.npz: timesteps, results, ep_lengths) and (b) the policy
weights of best_model.zip (every tensor of the policy state_dict), bit for bit.

    python docs/results/ppo_10_seeds/check_reproduction.py NEW_DIR OFFICIAL_EVALUATIONS_NPZ OFFICIAL_BEST_MODEL_ZIP
"""
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


def main():
    new_dir, off_eval, off_zip = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ok = True
    with np.load(new_dir / "evaluations.npz") as a, np.load(off_eval) as b:
        for k in ("timesteps", "results", "ep_lengths"):
            same = a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
            print(f"evaluations.npz[{k}]: {'IDENTICO' if same else 'DIFERENTE'}  shape {a[k].shape} vs {b[k].shape}")
            ok &= same
        print(f"   evaluaciones (media por evaluacion): {np.round(a['results'].mean(axis=1), 2).tolist()}")
    sa = PPO.load(new_dir / "best_model.zip", device="cpu").policy.state_dict()
    sb = PPO.load(off_zip, device="cpu").policy.state_dict()
    same_w = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    print(f"pesos de best_model.zip ({len(sa)} tensores): {'IDENTICOS' if same_w else 'DIFERENTES'}")
    ok &= same_w
    print("CONTROL OK" if ok else "CONTROL FALLA - detener")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
