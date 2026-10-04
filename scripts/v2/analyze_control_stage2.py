"""Analysis of stage 2 of the v2 control phase (docs/v2/ADDENDUM_CONTROL.md, sections 5 and 11).

Written before the test evaluation ran. Reads the single test evaluation
(docs/results/v2/control/test_stage2.json, from scripts/v2/evaluate_control_v2.py) and the 20
training folders; simulates nothing.

- P1 (sueno_lstm vs sueno_transformer, total return): seed-level Welch CI at 1 - 0.05/6. If it
  excludes 0, the arm with the higher mean is adopted; otherwise the LSTM (pre-registered tie-break).
- P4-P6 for the adopted arm against the best references fixed on validation (section 9):
  espera_mas_larga (total and B0) and min_verde_y_cambiar (C0). Significant only if the seed-level
  test has p < 0.05/6. P2-P3 (direct RL) are stage 3.
- Descriptive, per controller: return (total and per intersection), catastrophic episodes
  (< -3,600), real phase switches per signal, "blocked" (fewer than 3 switches per episode on
  average in ANY signal), the selection's best evaluation (timestep and mean real return on
  validation 21000-21004, from evaluations.npz), real steps (run_info.json) and agreement with
  each reference per signal (same states, --reference-agreement).

    python scripts/v2/analyze_control_stage2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.four_intersections import TRAFFIC_SIGNAL_IDS  # noqa: E402

RESULTS = ROOT_DIR / "docs" / "results" / "v2" / "control" / "test_stage2.json"
OUTPUT = ROOT_DIR / "docs" / "results" / "v2" / "control" / "test_stage2_analysis.json"
ARMS = {"sueno_lstm": "lstm", "sueno_transformer": "transformer"}
ALPHA = 0.05 / 6
BEST_REFERENCE = {"total": "espera_mas_larga", "B0": "espera_mas_larga", "C0": "min_verde_y_cambiar"}
PLANNED = {"P4": "total", "P5": "B0", "P6": "C0"}
BLOCKED_SWITCHES = 3.0
CATASTROPHIC = -3600.0
DATASET_TRANSITIONS = 9600


def find(comparisons: list[dict], a: str, b: str, metric: str) -> dict:
    return next(c for c in comparisons if c["a"] == a and c["b"] == b and c["metric"] == metric)


def per_controller(episodes: list[dict], arm: str) -> list[dict]:
    rows = []
    for label in sorted({e["seed_label"] for e in episodes if e["policy"] == arm}):
        eps = [e for e in episodes if e["policy"] == arm and e["seed_label"] == label]
        run_dir = ROOT_DIR / Path(label).parent
        with np.load(run_dir / "evaluations.npz") as ev:
            means = ev["results"].mean(axis=1)
            best = int(np.argmax(means))
            timesteps = ev["timesteps"].tolist()
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        switches = {ts: float(np.mean([e[f"switches_{ts}"] for e in eps])) for ts in TRAFFIC_SIGNAL_IDS}
        agree_keys = sorted(k for k in eps[0] if k.startswith("agree_"))
        rows.append({
            "controller": run_dir.name, "seed": info["seed"],
            "mean_total": float(np.mean([e["reward"] for e in eps])),
            **{f"mean_{ts}": float(np.mean([e[f"reward_{ts}"] for e in eps])) for ts in TRAFFIC_SIGNAL_IDS},
            "worst": float(min(e["reward"] for e in eps)),
            "catastrophic": int(sum(e["reward"] < CATASTROPHIC for e in eps)),
            "switches_mean": switches,
            "blocked_signals": [ts for ts, s in switches.items() if s < BLOCKED_SWITCHES],
            "blocked": any(s < BLOCKED_SWITCHES for s in switches.values()),
            "selection_best_timestep": timesteps[best], "selection_best_eval_number": best + 1,
            "selection_n_evals": len(timesteps), "selection_best_validation_return": float(means[best]),
            "selection_validation_returns": means.tolist(),
            "real_steps_total": info["real_steps_total"], "imagined_steps": info["imagined_steps"],
            "agreement": {k[len("agree_"):]: float(np.mean([e[k] for e in eps])) for k in agree_keys},
        })
    return rows


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    summary, comparisons, episodes = data["summary"], data["comparisons"], data["episodes"]

    p1 = find(comparisons, "sueno_lstm", "sueno_transformer", "total")
    lo, hi = p1["seed_level"]["ci"]
    excludes_zero = lo > 0 or hi < 0
    adopted = ("sueno_lstm" if p1["mean_diff"] > 0 else "sueno_transformer") if excludes_zero else "sueno_lstm"
    decision = {"comparison": p1, "ci_excludes_zero": excludes_zero, "adopted": adopted,
                "by": "significant difference" if excludes_zero else "pre-registered tie-break (LSTM)"}

    planned = {}
    for name, metric in PLANNED.items():
        for arm in ARMS:
            c = find(comparisons, arm, BEST_REFERENCE[metric], metric)
            planned.setdefault(name, {})[arm] = {**c, "significant": c["seed_level"]["p"] < ALPHA,
                                                 "role": "planned" if arm == adopted else "descriptive"}

    controllers = {arm: per_controller(episodes, arm) for arm in ARMS}
    real_steps = {arm: [c["real_steps_total"] for c in rows] for arm, rows in controllers.items()}
    accounting = {arm: {"selection_steps_per_seed": steps,
                        "unshared_per_seed": [DATASET_TRANSITIONS + s for s in steps],
                        "shared_over_10_seeds_per_seed": [DATASET_TRANSITIONS / 10 + s for s in steps]}
                  for arm, steps in real_steps.items()}
    result = {"alpha_bonferroni": ALPHA, "p1_decision": decision, "planned": planned,
              "controllers": controllers,
              "blocked_count": {arm: sum(c["blocked"] for c in rows) for arm, rows in controllers.items()},
              "interactions": accounting,
              "catastrophic": {name: s["catastrophic"] for name, s in summary.items()},
              "reference_means": {name: {m: summary[name][m]["mean"] for m in ("total", *TRAFFIC_SIGNAL_IDS)}
                                  for name in summary if name not in ARMS}}
    OUTPUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"P1 LSTM - Transformer: {p1['mean_diff']:+.1f}, IC {1 - ALPHA:.2%} [{lo:+.1f}, {hi:+.1f}], "
          f"p Welch {p1['seed_level']['p']:.3g}, t pareada p {p1['paired_t']['p']:.3g}, "
          f"Wilcoxon p {p1['wilcoxon']['p']:.3g} -> adoptado: {adopted} ({decision['by']})")
    for name, by_arm in planned.items():
        for arm, c in by_arm.items():
            sl = c["seed_level"]
            print(f"{name} [{c['role']}] {arm} - {c['b']} ({c['metric']}): {c['mean_diff']:+.1f}, "
                  f"p {sl['p']:.3g}, IC [{sl['ci'][0]:+.1f}, {sl['ci'][1]:+.1f}], t pareada p {c['paired_t']['p']:.3g}, "
                  f"Wilcoxon p {c['wilcoxon']['p']:.3g}, gana {c['a_wins_scenarios']}/24 -> "
                  f"{'significativo' if c['significant'] else 'no significativo'}")
    for arm, rows in controllers.items():
        print(f"\n{arm}: bloqueados {result['blocked_count'][arm]}/10")
        for c in rows:
            sw = " ".join(f"{ts}:{c['switches_mean'][ts]:4.1f}" for ts in TRAFFIC_SIGNAL_IDS)
            print(f"  {c['controller']:22s} total {c['mean_total']:9.1f} | " +
                  " ".join(f"{ts} {c[f'mean_{ts}']:8.1f}" for ts in TRAFFIC_SIGNAL_IDS) +
                  f" | cat {c['catastrophic']:2d} | cambios {sw} | {'BLOQUEADO ' + ','.join(c['blocked_signals']) if c['blocked'] else ''}"
                  f" | mejor eval {c['selection_best_eval_number']}/{c['selection_n_evals']} (t={c['selection_best_timestep']}) "
                  f"val {c['selection_best_validation_return']:.1f}")
    print(f"\n-> {OUTPUT}")


if __name__ == "__main__":
    main()
