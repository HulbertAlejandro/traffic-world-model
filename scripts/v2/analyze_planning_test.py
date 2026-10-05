"""Stage 2 of the planner pre-registration: analysis of the single test evaluation on 25000-25047
(docs/v2/ADDENDUM_PLANIFICACION.md, sections 6, 13, 15 and 17). Written before the test ran.
Simulates nothing.

The 12 planned comparisons (section 6), with P*_a the best arm fixed in section 15 (plan_ppo, H = 3,
for both architectures) and the best rules fixed there (espera_mas_larga total, cola_mas_larga B0,
min_verde_y_cambiar C0): Bonferroni alpha' = 0.05/12, CI at 1 - alpha'. Main test: Welch on the 10
seed/replica means (one-sample t against a rule); paired t and Wilcoxon by scenario are secondary
and marked as pseudoreplication. Every arm is reported, descriptively, with the same metrics:
mean and median, per intersection, catastrophic episodes (< -3,700, section 15), blocked
controllers and episodes (Phase 3 rules), switches per signal, ms per decision, agreement with the
references, seed means and which seeds dominate the variance. Real interactions with the two
accountings of section 13.

    python scripts/v2/analyze_planning_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
for path in (ROOT_DIR, ROOT_DIR / "scripts" / "v2"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from environments.four_intersections import TRAFFIC_SIGNAL_IDS  # noqa: E402
from scripts.v2 import evaluate_control_v2 as ev  # noqa: E402

RESULTS = ROOT_DIR / "docs" / "results" / "v2" / "planning" / "test"
OUTPUT = RESULTS.parent / "test_analysis.json"
SCENARIOS = list(range(25000, 25048))
THRESHOLD = -3700.0                         # section 15
ALPHA = 0.05 / 12
LEVEL = 1 - ALPHA
BEST_ARM = {"lstm": "plan_ppo_lstm", "transformer": "plan_ppo_transformer"}    # section 15
BEST_RULE = {"total": "espera_mas_larga", "B0": "cola_mas_larga", "C0": "min_verde_y_cambiar"}
PLANNERS = ("plan_solo_lstm", "plan_ppo_lstm", "plan_solo_transformer", "plan_ppo_transformer")
LEARNED = (*PLANNERS, "sueno_lstm", "sueno_transformer", "directo_10k", "directo_30k")
REFERENCES = ("fijo_2_3", "min_verde_y_cambiar", "cola_mas_larga", "max_presion", "espera_mas_larga")
BLOCKED = 3
# Real interactions per replica/seed (sections 8 and 13 of this addendum; ADDENDUM_CONTROL.md 4).
INTERACTIONS = {
    "planner (section 13, H and arm validated with replicas 0-2)": {"unshared": 21240, "shared": 6552},
    "sueno (Phase 3)": {"unshared": 12600, "shared": 3960},
    "directo_10k": {"unshared": 13240, "shared": 13240},
    "directo_30k": {"unshared": 39208, "shared": 39208},
}


def planned() -> list[tuple[str, str, str, str]]:
    out = []
    for arch, best in BEST_ARM.items():
        out += [(f"Q1_{arch}", best, f"sueno_{arch}", "total"), (f"Q2_{arch}", best, "directo_30k", "total"),
                (f"Q3_{arch}", best, "directo_10k", "total"), (f"Q4_{arch}", best, BEST_RULE["total"], "total"),
                (f"Q5_{arch}", best, BEST_RULE["B0"], "B0"), (f"Q6_{arch}", best, BEST_RULE["C0"], "C0")]
    return out


def describe(episodes: list[dict], summary_entry: dict) -> dict:
    totals = np.array([e["reward"] for e in episodes])
    labels = summary_entry["seed_labels"]
    seed_means = summary_entry["total"]["seed_means"]
    switches = {}
    for label in labels:
        rows = [e for e in episodes if e["seed_label"] == label]
        switches[label] = {ts: float(np.mean([e[f"switches_{ts}"] for e in rows])) for ts in TRAFFIC_SIGNAL_IDS}
    dev = np.array(seed_means) - np.mean(seed_means)
    share = (dev ** 2) / (dev ** 2).sum() if len(dev) > 1 and (dev ** 2).sum() > 0 else np.zeros(len(dev))
    agree = sorted(k for k in episodes[0] if k.startswith("agree_"))
    out = {
        "mean": float(np.mean(seed_means)), "median": float(np.median(totals)), "n_episodes": len(episodes),
        "per_signal": {ts: summary_entry[ts]["mean"] for ts in TRAFFIC_SIGNAL_IDS},
        "catastrophic": int((totals < THRESHOLD).sum()),
        "blocked_controllers": [label for label, sw in switches.items() if min(sw.values()) < BLOCKED],
        "blocked_episodes": int(sum(min(e[f"switches_{ts}"] for ts in TRAFFIC_SIGNAL_IDS) < BLOCKED for e in episodes)),
        "switches_mean": {ts: float(np.mean([e[f"switches_{ts}"] for e in episodes])) for ts in TRAFFIC_SIGNAL_IDS},
        "min_switches_per_controller": {label: min(sw.values()) for label, sw in switches.items()},
        "seed_means": dict(zip(labels, seed_means)),
        "variance_share_by_seed": dict(zip(labels, share.tolist())),
        "agreement": {k[len("agree_"):]: float(np.mean([e[k] for e in episodes])) for k in agree},
    }
    if "decision_seconds_mean" in episodes[0]:
        out["ms_per_decision"] = 1000 * float(np.mean([e["decision_seconds_mean"] for e in episodes]))
    return out


def main() -> None:
    episodes, names = [], []
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["arguments"]["scenarios"] != SCENARIOS:
            raise SystemExit(f"{path} is not on 25000-25047")
        episodes += data["episodes"]
        names += [p["name"] for p in data["arguments"]["policies"] if p["name"] not in names]
    missing = (set(LEARNED) | set(REFERENCES)) - set(names)
    if missing:
        raise SystemExit(f"missing policies: {sorted(missing)}")
    summary = ev.summarize(episodes, names, SCENARIOS, THRESHOLD)
    for name in LEARNED:
        if summary[name]["n_seeds"] != 10:
            raise SystemExit(f"{name} has {summary[name]['n_seeds']} seeds, expected 10")
    arms = {name: describe([e for e in episodes if e["policy"] == name], summary[name]) for name in names}
    comparisons = {}
    for label, a, b, metric in planned():
        c = ev.compare(summary, a, b, metric, LEVEL)
        comparisons[label] = {**c, "significant": c["seed_level"]["p"] < ALPHA,
                              "secondary_note": "paired t and Wilcoxon by scenario: pseudoreplication"}
    descriptive = {}
    for arm in PLANNERS:
        if arm in BEST_ARM.values():
            continue
        arch = arm.split("_")[-1]
        for target, metric in ((f"sueno_{arch}", "total"), ("directo_30k", "total"), ("directo_10k", "total"),
                               (BEST_RULE["total"], "total"), (BEST_RULE["B0"], "B0"), (BEST_RULE["C0"], "C0")):
            descriptive[f"{arm} - {target} ({metric})"] = ev.compare(summary, arm, target, metric, LEVEL)
    result = {"alpha_bonferroni": ALPHA, "level": LEVEL, "threshold": THRESHOLD, "best_arm": BEST_ARM,
              "best_rule": BEST_RULE, "planned": comparisons, "descriptive_other_arms": descriptive,
              "arms": arms, "interactions": INTERACTIONS}
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
    print("\nVarianza por semilla (la que mas aporta):")
    for name in LEARNED:
        share = arms[name]["variance_share_by_seed"]
        top = max(share, key=share.get)
        print(f"  {name:22s} {top}: {share[top]:.0%} (media {arms[name]['seed_means'][top]:.1f}); "
              f"bloqueados {arms[name]['blocked_controllers'] or 'ninguno'}")
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
