"""Paso 4 of the planner pre-registration: apply the selection rules to the validation results
(docs/v2/ADDENDUM_PLANIFICACION.md, sections 3, 5, 6 and 13). Written before the validation ran.
Simulates nothing.

Rules, unchanged:
- H per arm: the H with the highest mean validation return (lowest cost), the mean of the replica
  means of replicas 0, 1 and 2; exact tie -> the smaller H.
- Best arm per architecture: plan_solo vs plan_ppo, each at its H, the higher mean; exact tie -> plan_solo.
- Catastrophic threshold: 1.5 x the worst fijo_2_3 return on 24000-24023, rounded down to the hundred.
- Best rule: the highest mean validation return, separately for the total, B0 and C0.

Descriptive, per arm and H: mean and median, per intersection, catastrophic episodes, blocked
controllers (mean < 3 switches per episode in some signal) and blocked episodes (< 3 in some signal),
switches per signal, time per decision and per episode, agreement with the references; next to the
20 Phase 3 dream PPO on the same seeds (docs/results/v2/dream_alignment/validation_v21_phase3_*.json)
and the 5 rules.

    python scripts/v2/analyze_planning_validation.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.four_intersections import TRAFFIC_SIGNAL_IDS  # noqa: E402

RESULTS = ROOT_DIR / "docs" / "results" / "v2" / "planning" / "validation"
PHASE3 = ROOT_DIR / "docs" / "results" / "v2" / "dream_alignment"
OUTPUT = RESULTS.parent / "validation_analysis.json"
ARMS = [f"plan_{cont}_{arch}" for arch in ("lstm", "transformer") for cont in ("solo", "ppo")]
HORIZONS = (3, 5, 7)
REFERENCES = ("fijo_2_3", "min_verde_y_cambiar", "cola_mas_larga", "max_presion", "espera_mas_larga")
BLOCKED = 3
SCENARIOS = list(range(24000, 24024))


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["arguments"]["scenarios"] != SCENARIOS:
        raise SystemExit(f"{path} is not on 24000-24023")
    return data["episodes"]


def describe(episodes: list[dict], threshold: float) -> dict:
    totals = np.array([e["reward"] for e in episodes])
    labels = sorted({e["seed_label"] for e in episodes})
    controllers = {}
    for label in labels:
        rows = [e for e in episodes if e["seed_label"] == label]
        switches = {ts: float(np.mean([e[f"switches_{ts}"] for e in rows])) for ts in TRAFFIC_SIGNAL_IDS}
        controllers[label] = {"mean": float(np.mean([e["reward"] for e in rows])), "switches_mean": switches,
                              "blocked_signals": [ts for ts, v in switches.items() if v < BLOCKED]}
    agree = sorted(k for k in episodes[0] if k.startswith("agree_"))
    out = {
        "n_episodes": len(episodes), "replica_means": [controllers[label]["mean"] for label in labels],
        "mean": float(np.mean([controllers[label]["mean"] for label in labels])),
        "median": float(np.median(totals)),
        "per_signal": {ts: float(np.mean([e[f"reward_{ts}"] for e in episodes])) for ts in TRAFFIC_SIGNAL_IDS},
        "catastrophic": int((totals < threshold).sum()),
        "blocked_controllers": [label for label, c in controllers.items() if c["blocked_signals"]],
        "blocked_episodes": int(sum(min(e[f"switches_{ts}"] for ts in TRAFFIC_SIGNAL_IDS) < BLOCKED for e in episodes)),
        "switches_mean": {ts: float(np.mean([e[f"switches_{ts}"] for e in episodes])) for ts in TRAFFIC_SIGNAL_IDS},
        "episode_seconds": float(np.mean([e["seconds"] for e in episodes])),
        "agreement": {k[len("agree_"):]: float(np.mean([e[k] for e in episodes])) for k in agree},
        "controllers": controllers,
    }
    if "decision_seconds_mean" in episodes[0]:
        out["decision_seconds"] = float(np.mean([e["decision_seconds_mean"] for e in episodes]))
    return out


def main() -> None:
    refs = load(RESULTS / "references.json")
    worst_fixed = min(e["reward"] for e in refs if e["policy"] == "fijo_2_3")
    threshold = math.floor(1.5 * worst_fixed / 100) * 100
    rules = {r: describe([e for e in refs if e["policy"] == r], threshold) for r in REFERENCES}
    best_rule = {m: max(REFERENCES, key=lambda r: rules[r]["mean"] if m == "total" else rules[r]["per_signal"][m])
                 for m in ("total", "B0", "C0")}

    planners = {arm: {h: describe(load(RESULTS / f"{arm}_H{h}.json"), threshold) for h in HORIZONS} for arm in ARMS}
    chosen_h = {}
    for arm, by_h in planners.items():
        best = max(by_h[h]["mean"] for h in HORIZONS)
        chosen_h[arm] = min(h for h in HORIZONS if by_h[h]["mean"] == best)  # exact tie -> smaller H
    best_arm = {}
    for arch in ("lstm", "transformer"):
        solo, ppo = planners[f"plan_solo_{arch}"][chosen_h[f"plan_solo_{arch}"]], planners[f"plan_ppo_{arch}"][chosen_h[f"plan_ppo_{arch}"]]
        best_arm[arch] = f"plan_ppo_{arch}" if ppo["mean"] > solo["mean"] else f"plan_solo_{arch}"  # tie -> solo

    phase3 = {}
    for arch in ("lstm", "transformer"):
        eps = load(PHASE3 / f"validation_v21_phase3_{arch}.json")
        phase3[f"sueno_{arch}_10_semillas"] = describe(eps, threshold)
        phase3[f"sueno_{arch}_semillas_0_1_2"] = describe(
            [e for e in eps if any(e["seed_label"].endswith(f"_s{s}/best_model.zip") for s in (0, 1, 2))], threshold)

    selected = {"threshold": threshold, "worst_fijo_2_3": worst_fixed, "best_rule": best_rule,
                "best_rule_values": {m: (rules[r]["mean"] if m == "total" else rules[r]["per_signal"][m])
                                     for m, r in best_rule.items()},
                "chosen_H": chosen_h, "best_arm": best_arm,
                "best_arm_values": {arch: planners[a][chosen_h[a]]["mean"] for arch, a in best_arm.items()}}
    OUTPUT.write_text(json.dumps({"selected": selected, "planners": {a: {str(h): v for h, v in d.items()}
                                                                     for a, d in planners.items()},
                                  "rules": rules, "phase3_dream": phase3}, indent=1), encoding="utf-8")

    print("SELECCION (reglas pre-registradas):")
    print(f"  umbral de catastroficos: {threshold:.0f} (peor fijo_2_3 {worst_fixed:.1f})")
    print(f"  mejor regla: {best_rule} -> {selected['best_rule_values']}")
    print(f"  H por brazo: {chosen_h}")
    print(f"  mejor brazo por arquitectura: {best_arm}")
    header = (f"{'brazo':28s} | {'media':>9s} | {'mediana':>8s} | " + " | ".join(f"{t:>7s}" for t in TRAFFIC_SIGNAL_IDS)
              + " | cat. | ep.bloq | ctrl bloq | cambios A0/B0/C0/D0 | ms/dec | s/ep")
    print("\n" + header)

    def row(name, s):
        dec = f"{1000 * s['decision_seconds']:6.0f}" if "decision_seconds" in s else "     -"
        print(f"{name:28s} | {s['mean']:9.1f} | {s['median']:8.1f} | " + " | ".join(f"{s['per_signal'][t]:7.1f}" for t in TRAFFIC_SIGNAL_IDS)
              + f" | {s['catastrophic']:3d}/{s['n_episodes']} | {s['blocked_episodes']:3d}/{s['n_episodes']} | "
              f"{len(s['blocked_controllers'])}/{len(s['replica_means'])} | "
              + " / ".join(f"{s['switches_mean'][t]:4.1f}" for t in TRAFFIC_SIGNAL_IDS) + f" | {dec} | {s['episode_seconds']:5.1f}")

    for arm, by_h in planners.items():
        for h in HORIZONS:
            row(f"{arm}_H{h}{' *' if chosen_h[arm] == h else ''}", by_h[h])
    for name, s in phase3.items():
        row(name, s)
    for r in REFERENCES:
        row(r, rules[r])
    print("\nBloqueados:")
    for arm, by_h in planners.items():
        for h in HORIZONS:
            for label in by_h[h]["blocked_controllers"]:
                print(f"  {arm}_H{h} {label}: {by_h[h]['controllers'][label]['switches_mean']}")
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
