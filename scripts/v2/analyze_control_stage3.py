"""Analysis of stage 3 of the v2 control phase (docs/v2/ADDENDUM_CONTROL.md, section 12).

Written before the stage-3 test evaluation ran. Simulates nothing. Combines the stage-2 test
episodes (the 20 dream controllers and the 5 references, test_stage2.json; not re-evaluated) with
the stage-3 ones (the 20 direct-RL controllers, test_stage3.json): same 24 scenarios, deterministic
episodes per scenario.

The 8 planned comparisons of section 12 (P1, P2a/b, P3a/b, P4-P6), Bonferroni alpha' = 0.05/8, CI
at 99.375%. Main test: Welch on the seed means (one-sample t against a deterministic reference);
paired t and Wilcoxon by scenario are secondary and labelled as pseudoreplication with respect to
the method. Descriptive per direct controller: return (total and per intersection), catastrophic
episodes, phase switches and "blocked" (stage 2's rule), selection's best evaluation, real steps.
Real interactions with the two accountings of section 4.

    python scripts/v2/analyze_control_stage3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "scripts" / "v2") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts" / "v2"))

from environments.four_intersections import TRAFFIC_SIGNAL_IDS  # noqa: E402
from scripts.v2 import evaluate_control_v2 as ev  # noqa: E402
from scripts.v2.analyze_control_stage2 import DATASET_TRANSITIONS, per_controller  # noqa: E402

RESULTS_DIR = ROOT_DIR / "docs" / "results" / "v2" / "control"
STAGE2, STAGE3 = RESULTS_DIR / "test_stage2.json", RESULTS_DIR / "test_stage3.json"
OUTPUT = RESULTS_DIR / "test_stage3_analysis.json"
ALPHA = 0.05 / 8
LEVEL = 1 - ALPHA
PLANNED = (
    ("P1", "sueno_lstm", "sueno_transformer", "total"),
    ("P2a", "sueno_lstm", "directo_10k", "total"),
    ("P2b", "sueno_transformer", "directo_10k", "total"),
    ("P3a", "sueno_lstm", "directo_30k", "total"),
    ("P3b", "sueno_transformer", "directo_30k", "total"),
    ("P4", "sueno_lstm", "espera_mas_larga", "total"),
    ("P5", "sueno_lstm", "espera_mas_larga", "B0"),
    ("P6", "sueno_lstm", "min_verde_y_cambiar", "C0"),
)
DREAM_ARMS = ("sueno_lstm", "sueno_transformer")
DIRECT_ARMS = ("directo_10k", "directo_30k")


def main() -> None:
    stage2 = json.loads(STAGE2.read_text(encoding="utf-8"))
    stage3 = json.loads(STAGE3.read_text(encoding="utf-8"))
    scenarios = stage2["arguments"]["scenarios"]
    if stage3["arguments"]["scenarios"] != scenarios:
        raise SystemExit("stage 2 and stage 3 were not evaluated on the same scenarios")
    episodes = stage2["episodes"] + stage3["episodes"]
    names = [p["name"] for p in stage2["arguments"]["policies"]] + [p["name"] for p in stage3["arguments"]["policies"]]
    summary = ev.summarize(episodes, names, scenarios, ev.CATASTROPHIC_THRESHOLD)

    planned = {}
    for label, a, b, metric in PLANNED:
        c = ev.compare(summary, a, b, metric, LEVEL)
        planned[label] = {**c, "significant": c["seed_level"]["p"] < ALPHA,
                          "secondary_note": "paired t and Wilcoxon: pseudoreplication with respect to the method"}

    direct = {arm: per_controller(episodes, arm) for arm in DIRECT_ARMS}
    dream_unshared = DATASET_TRANSITIONS + 3000
    dream_shared = DATASET_TRANSITIONS / 10 + 3000
    interactions = {"dream_per_seed": {"unshared": dream_unshared, "shared_over_10_seeds": dream_shared,
                                       "selection_steps": 3000}}
    for arm, rows in direct.items():
        steps = [c["real_steps_total"] for c in rows]
        interactions[arm] = {"real_steps_per_seed": steps,
                             "ratio_vs_dream_unshared": [min(steps) / dream_unshared, max(steps) / dream_unshared],
                             "ratio_vs_dream_shared": [min(steps) / dream_shared, max(steps) / dream_shared]}
    result = {"alpha_bonferroni": ALPHA, "level": LEVEL, "planned": planned, "direct_controllers": direct,
              "blocked_count": {arm: sum(c["blocked"] for c in rows) for arm, rows in direct.items()},
              "interactions": interactions,
              "summary": {n: {"mean": {m: summary[n][m]["mean"] for m in ("total", *TRAFFIC_SIGNAL_IDS)},
                              "median_total": float(np.median([e["reward"] for e in episodes if e["policy"] == n])),
                              "seed_means_total": summary[n]["total"]["seed_means"],
                              "catastrophic": summary[n]["catastrophic"], "n_episodes": summary[n]["n_episodes"],
                              "switches_mean": summary[n]["switches_mean"]} for n in names}}
    OUTPUT.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")

    print(f"{'Politica':20s} | {'total':>9s} | {'mediana':>8s} | " + " | ".join(f"{t:>8s}" for t in TRAFFIC_SIGNAL_IDS)
          + " | catastr.")
    for n in names:
        s = result["summary"][n]
        print(f"{n:20s} | {s['mean']['total']:9.1f} | {s['median_total']:8.1f} | "
              + " | ".join(f"{s['mean'][t]:8.1f}" for t in TRAFFIC_SIGNAL_IDS) + f" | {s['catastrophic']}/{s['n_episodes']}")
    print(f"\nComparaciones planificadas (alfa' = {ALPHA:.5f}, IC {LEVEL:.3%}); principal: nivel semilla")
    for label, c in planned.items():
        sl = c["seed_level"]
        print(f"{label:4s} {c['a']} - {c['b']} ({c['metric']}): {c['mean_diff']:+.1f}; {sl['test']} p={sl['p']:.3g} "
              f"IC [{sl['ci'][0]:+.1f}, {sl['ci'][1]:+.1f}] -> {'SIGNIFICATIVO' if c['significant'] else 'no significativo'}"
              f" | secundarios (pseudorreplicacion): t pareada p={c['paired_t']['p']:.3g}, Wilcoxon p={c['wilcoxon']['p']:.3g},"
              f" gana {c['a_wins_scenarios']}/24")
    for arm, rows in direct.items():
        print(f"\n{arm}: bloqueados {result['blocked_count'][arm]}/10; pasos reales {interactions[arm]['real_steps_per_seed']}")
        for c in rows:
            sw = " ".join(f"{t}:{c['switches_mean'][t]:4.1f}" for t in TRAFFIC_SIGNAL_IDS)
            print(f"  {c['controller']:16s} total {c['mean_total']:9.1f} | cat {c['catastrophic']:2d} | cambios {sw} "
                  f"{'BLOQUEADO ' + ','.join(c['blocked_signals']) if c['blocked'] else ''}| mejor eval "
                  f"{c['selection_best_eval_number']}/{c['selection_n_evals']} (t={c['selection_best_timestep']}) "
                  f"val {c['selection_best_validation_return']:.1f}")
    print(f"\nInteracciones: {json.dumps(interactions, default=float)}")
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
