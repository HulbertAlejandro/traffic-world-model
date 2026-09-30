"""Applies the pre-registered winner criterion (MANIFEST.md) to the sweep_z{K}.json reports."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
rows = []
for k in (4, 8, 12, 16, 20):
    report = json.loads((HERE / f"sweep_z{k}.json").read_text(encoding="utf-8"))
    per_seed = report["per_seed"]
    assert [s["seed"] for s in per_seed] == [0, 1, 2]
    rows.append({"latent_dim": k, "median_of_seed_reductions": float(np.median([s["median_reduction"] for s in per_seed])),
                 "z_wins_pairs": report["z_wins_total"], "per_seed": per_seed})
# 1) highest median of per-seed reductions; 2) more pairs won; 3) smaller latent_dim
winner = sorted(rows, key=lambda r: (-r["median_of_seed_reductions"], -r["z_wins_pairs"], r["latent_dim"]))[0]
for r in rows:
    seeds = ", ".join(f"s{s['seed']} {100 * s['median_reduction']:+.1f}% ({s['z_wins_horizons']}/10)" for s in r["per_seed"])
    print(f"latent_dim {r['latent_dim']:>2}: mediana {100 * r['median_of_seed_reductions']:+.1f}% | pares {r['z_wins_pairs']}/30 | {seeds}")
print(f"GANADOR (criterio del MANIFEST): latent_dim = {winner['latent_dim']}")
(HERE / "sweep_summary.json").write_text(json.dumps({"rows": rows, "winner": winner["latent_dim"]}, indent=2), encoding="utf-8")
