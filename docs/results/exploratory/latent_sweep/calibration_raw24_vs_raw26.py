"""Informational (MANIFEST): raw-24 LSTM vs official raw-26 LSTM, seeds 0-4, same reduction metric."""
import json
from pathlib import Path

import numpy as np
import torch

import sweep

ROOT, W, HERE = sweep.ROOT, sweep.WORK, Path(__file__).parent
rows = []
for s in range(5):
    r24, _ = sweep.cmp0.evaluate(W / f"raw24/seed{s}/world_model_best.pt", W / "data/test_raw_seq.npz", torch.device("cpu"))
    r26, _ = sweep.cmp0.evaluate(ROOT / f"models/checkpoints/exp0_multiseed_300ep/raw/seed{s}/world_model_raw_best.pt",
                                 ROOT / "datasets/processed/test_raw_seq.npz", torch.device("cpu"))
    red = np.array([(r26[h]["model_reward_mse"] - r24[h]["model_reward_mse"]) / r26[h]["model_reward_mse"] for h in range(1, 11)])
    rows.append({"seed": s, "median_reduction_raw24_vs_raw26": float(np.median(red)), "raw24_better_horizons": int((red > 0).sum()),
                 "h1": [r24[1]["model_reward_mse"], r26[1]["model_reward_mse"]], "h10": [r24[10]["model_reward_mse"], r26[10]["model_reward_mse"]]})
    print(f"seed {s}: (raw26 - raw24)/raw26 mediana {100 * np.median(red):+.1f}%, raw24 mejor en {(red > 0).sum()}/10 | "
          f"h1 {r24[1]['model_reward_mse']:.1f} vs {r26[1]['model_reward_mse']:.1f} | h10 {r24[10]['model_reward_mse']:.1f} vs {r26[10]['model_reward_mse']:.1f}")
(HERE / "calibration_raw24_vs_raw26.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
