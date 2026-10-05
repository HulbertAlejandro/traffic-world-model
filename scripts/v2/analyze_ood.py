"""Analysis of the final confirmation on OOD 23000-23029 (docs/v2/ADDENDUM_OOD.md). Written before the
OOD evaluation ran. Simulates nothing.

- The 12 pre-registered comparisons O1-O6 per architecture (the new test's Q1-Q6, same targets),
  Bonferroni alpha' = 0.05/12, CI at 1 - alpha'; main test Welch on the 10 seed means (one-sample t
  against a rule); paired t and Wilcoxon by scenario secondary (pseudoreplication).
- Every policy described with the new test's metrics (analyze_planning_test.describe), the
  catastrophic threshold -3,700 and the best rules fixed on validation_v21.
- The pre-registered reading of section 3, applied literally: O2_transformer positive and
  significant -> confirms; positive and not significant -> partial confirmation; negative (or
  significant against) -> contradicts.
- Reproducibility: on the OOD scenarios where the dataset used fijo_2_3 or cola_mas_larga, the
  evaluated return must equal the manifest's.

    python scripts/v2/analyze_ood.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
for path in (ROOT_DIR, ROOT_DIR / "scripts" / "v2"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from environments.four_intersections import TRAFFIC_SIGNAL_IDS  # noqa: E402
from scripts.v2 import evaluate_control_v2 as ev  # noqa: E402
from scripts.v2.analyze_planning_test import (  # noqa: E402
    ALPHA,
    BEST_ARM,
    BEST_RULE,
    INTERACTIONS,
    LEVEL,
    REFERENCES,
    THRESHOLD,
    describe,
    planned,
)

RESULTS = ROOT_DIR / "docs" / "results" / "v2" / "ood"
OUTPUT = RESULTS.parent / "ood_analysis.json"
MANIFEST = ROOT_DIR / "docs" / "results" / "v2" / "dataset" / "manifest.json"
SCENARIOS = list(range(23000, 23030))
LEARNED = ("plan_ppo_lstm", "plan_ppo_transformer", "sueno_lstm", "sueno_transformer", "directo_10k", "directo_30k")


def reading(c: dict) -> str:
    """Section 3 of the addendum, for O2_transformer."""
    if c["mean_diff"] > 0 and c["significant"]:
        return "confirma"
    if c["mean_diff"] > 0:
        return "confirma parcialmente"
    return "contradice"


def reproducibility(episodes: list[dict]) -> dict:
    manifest = {e["seed"]: e for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["episodes"] if e["split"] == "ood"}
    checked, diffs = 0, []
    for e in episodes:
        m = manifest.get(e["scenario_seed"])
        if m and m["policy"] == e["policy"] and e["policy"] in ("fijo_2_3", "cola_mas_larga"):
            checked += 1
            diffs.append({"scenario": e["scenario_seed"], "policy": e["policy"], "manifest": m["return"],
                          "evaluated": e["reward"], "diff": e["reward"] - m["return"]})
    return {"checked": checked, "max_abs_diff": max((abs(d["diff"]) for d in diffs), default=None),
            "mismatches": [d for d in diffs if abs(d["diff"]) > 1e-6]}


def main() -> None:
    episodes, names = [], []
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["arguments"]["scenarios"] != SCENARIOS:
            raise SystemExit(f"{path} is not on 23000-23029")
        episodes += data["episodes"]
        names += [p["name"] for p in data["arguments"]["policies"] if p["name"] not in names]
    missing = (set(LEARNED) | set(REFERENCES)) - set(names)
    extra = set(names) - set(LEARNED) - set(REFERENCES)
    if missing or extra:
        raise SystemExit(f"policies differ from the pre-registration: missing {sorted(missing)}, extra {sorted(extra)}")
    summary = ev.summarize(episodes, names, SCENARIOS, THRESHOLD)
    for name in LEARNED:
        if summary[name]["n_seeds"] != 10:
            raise SystemExit(f"{name} has {summary[name]['n_seeds']} seeds, expected 10")
    arms = {name: describe([e for e in episodes if e["policy"] == name], summary[name]) for name in names}
    comparisons = {}
    for label, a, b, metric in planned():
        c = ev.compare(summary, a, b, metric, LEVEL)
        comparisons[label.replace("Q", "O", 1)] = {**c, "significant": c["seed_level"]["p"] < ALPHA,
                                                   "secondary_note": "paired t and Wilcoxon by scenario: pseudoreplication"}
    verdict = reading(comparisons["O2_transformer"])
    repro = reproducibility(episodes)
    result = {"alpha_bonferroni": ALPHA, "level": LEVEL, "threshold": THRESHOLD, "best_arm": BEST_ARM,
              "best_rule": BEST_RULE, "planned": comparisons, "pre_registered_reading_O2_transformer": verdict,
              "reproducibility": repro, "arms": arms, "interactions": INTERACTIONS}
    OUTPUT.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")

    print(f"{'politica':22s} | {'media':>9s} | {'mediana':>8s} | " + " | ".join(f"{t:>7s}" for t in TRAFFIC_SIGNAL_IDS)
          + " | catastr. | ep.bloq | ctrl bloq | cambios A0/B0/C0/D0 | ms/dec")
    for name in (*LEARNED, *REFERENCES):
        s = arms[name]
        ms = f"{s['ms_per_decision']:6.1f}" if "ms_per_decision" in s else "     -"
        print(f"{name:22s} | {s['mean']:9.1f} | {s['median']:8.1f} | " + " | ".join(f"{s['per_signal'][t]:7.1f}" for t in TRAFFIC_SIGNAL_IDS)
              + f" | {s['catastrophic']:3d}/{s['n_episodes']} | {s['blocked_episodes']:3d}/{s['n_episodes']} | "
              f"{len(s['blocked_controllers']):2d} | " + " / ".join(f"{s['switches_mean'][t]:4.1f}" for t in TRAFFIC_SIGNAL_IDS) + f" | {ms}")
    print(f"\nComparaciones planificadas (alfa' = {ALPHA:.5f}, IC {LEVEL:.3%}); principal: nivel semilla")
    for label, c in comparisons.items():
        sl = c["seed_level"]
        print(f"{label:15s} {c['a']} - {c['b']} ({c['metric']}): {c['mean_diff']:+.1f}; {sl['test']} p={sl['p']:.3g} "
              f"IC [{sl['ci'][0]:+.1f}, {sl['ci'][1]:+.1f}] -> {'SIGNIFICATIVO' if c['significant'] else 'no significativo'}"
              f" | secundarios (pseudorreplicacion): t pareada p={c['paired_t']['p']:.3g}, Wilcoxon p={c['wilcoxon']['p']:.3g}, "
              f"gana {c['a_wins_scenarios']}/{c['n_scenarios']}")
    print(f"\nLectura pre-registrada (O2_transformer): {verdict}")
    print(f"Reproducibilidad contra el manifiesto: {repro['checked']} episodios, diferencia maxima {repro['max_abs_diff']}, "
          f"discrepancias {len(repro['mismatches'])}")
    print("\nVarianza por semilla (la que mas aporta):")
    for name in LEARNED:
        share = arms[name]["variance_share_by_seed"]
        top = max(share, key=share.get)
        print(f"  {name:22s} {top}: {share[top]:.0%} (media {arms[name]['seed_means'][top]:.1f}); "
              f"bloqueados {arms[name]['blocked_controllers'] or 'ninguno'}")
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
