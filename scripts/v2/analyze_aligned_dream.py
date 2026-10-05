"""Analysis of v2.1-0, the corrected dream (docs/v2/ADDENDUM_SUENO_CORREGIDO.md). Descriptive.

Written before the validation_v21 evaluation ran. Reads the four evaluation files (Phase 3 and
corrected PPO, LSTM and Transformer, 24000-24023) and simulates nothing:

- per arm: mean and median total return, per-intersection means, catastrophic episodes (< -3,600),
  blocked controllers (mean < 3 switches per episode in some signal) and blocked episodes (< 3 in
  some signal), switches per signal, agreement with the references;
- per architecture, corrected - Phase 3: Welch on the 10 seed means with a 95% CI (uncorrected);
- the pre-registered "important change" criteria (section 5) and, from them, which PPO plan_ppo uses
  (ADDENDUM_PLANIFICACION.md, section 13).

    python scripts/v2/analyze_aligned_dream.py
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

import analyze_lstm_vs_transformer as stats  # noqa: E402
from environments.four_intersections import TRAFFIC_SIGNAL_IDS  # noqa: E402

RESULTS_DIR = ROOT_DIR / "docs" / "results" / "v2" / "dream_alignment"
FILES = {(arch, version): RESULTS_DIR / f"validation_v21_{version}_{arch}.json"
         for arch in ("lstm", "transformer") for version in ("phase3", "aligned")}
OUTPUT = RESULTS_DIR / "validation_v21_analysis.json"
CATASTROPHIC = -3600.0
BLOCKED_SWITCHES = 3
MIN_COUNT = 10           # section 5: halving only counts when Phase 3 had at least 10 such episodes
PLAN_PPO_DIRS = {"phase3": "models/checkpoints/v2/control/dream_{arch}_s<i>",
                 "aligned": "models/checkpoints/v2/control_aligned/dream_{arch}_s<i>"}


def arm_summary(episodes: list[dict]) -> dict:
    totals = np.array([e["reward"] for e in episodes])
    labels = sorted({e["seed_label"] for e in episodes})
    per_ctrl = {}
    for label in labels:
        rows = [e for e in episodes if e["seed_label"] == label]
        switches = {ts: float(np.mean([e[f"switches_{ts}"] for e in rows])) for ts in TRAFFIC_SIGNAL_IDS}
        per_ctrl[label] = {"mean": float(np.mean([e["reward"] for e in rows])), "switches_mean": switches,
                           "blocked": any(v < BLOCKED_SWITCHES for v in switches.values()),
                           "catastrophic": int(sum(e["reward"] < CATASTROPHIC for e in rows))}
    agree_keys = sorted(k for k in episodes[0] if k.startswith("agree_"))
    return {
        "n_episodes": len(episodes), "mean": float(totals.mean()), "median": float(np.median(totals)),
        "per_signal": {ts: float(np.mean([e[f"reward_{ts}"] for e in episodes])) for ts in TRAFFIC_SIGNAL_IDS},
        "catastrophic": int((totals < CATASTROPHIC).sum()),
        "blocked_episodes": int(sum(min(e[f"switches_{ts}"] for ts in TRAFFIC_SIGNAL_IDS) < BLOCKED_SWITCHES
                                    for e in episodes)),
        "blocked_controllers": int(sum(c["blocked"] for c in per_ctrl.values())),
        "switches_mean": {ts: float(np.mean([e[f"switches_{ts}"] for e in episodes])) for ts in TRAFFIC_SIGNAL_IDS},
        "agreement": {k[len("agree_"):]: float(np.mean([e[k] for e in episodes])) for k in agree_keys},
        "seed_means": [per_ctrl[label]["mean"] for label in labels], "controllers": per_ctrl,
    }


def important(old: dict, new: dict, welch: dict) -> dict:
    halved = lambda key: old[key] >= MIN_COUNT and new[key] < 0.5 * old[key]  # noqa: E731
    criteria = {"blocked_episodes_halved": halved("blocked_episodes"),
                "catastrophic_halved": halved("catastrophic"),
                "return_ci_positive": welch["ci95"][0] > 0}
    return {"criteria": criteria, "important": any(criteria.values())}


def main() -> None:
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in FILES.items()}
    scenarios = {tuple(d["arguments"]["scenarios"]) for d in data.values()}
    if scenarios != {tuple(range(24000, 24024))}:
        raise SystemExit(f"the four evaluations must use 24000-24023, got {scenarios}")
    arms = {key: arm_summary(d["episodes"]) for key, d in data.items()}
    result = {"arms": {f"{v}_{a}": s for (a, v), s in arms.items()}, "comparisons": {}, "plan_ppo": {}}
    for arch in ("lstm", "transformer"):
        old, new = arms[(arch, "phase3")], arms[(arch, "aligned")]
        w = stats.welch(np.array(new["seed_means"]), np.array(old["seed_means"]), 0.95)
        verdict = important(old, new, w)
        result["comparisons"][arch] = {"aligned_minus_phase3": w, **verdict}
        version = "aligned" if verdict["important"] else "phase3"
        result["plan_ppo"][arch] = {"uses": version, "checkpoints": PLAN_PPO_DIRS[version].format(arch=arch)}
    OUTPUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"{'brazo':22s} | {'media':>9s} | {'mediana':>8s} | " + " | ".join(f"{t:>8s}" for t in TRAFFIC_SIGNAL_IDS)
          + " | catastr. | ep. bloq. | ctrl bloq. | cambios A0/B0/C0/D0")
    for (arch, version), s in arms.items():
        print(f"{version + '_' + arch:22s} | {s['mean']:9.1f} | {s['median']:8.1f} | "
              + " | ".join(f"{s['per_signal'][t]:8.1f}" for t in TRAFFIC_SIGNAL_IDS)
              + f" | {s['catastrophic']:3d}/{s['n_episodes']} | {s['blocked_episodes']:3d}/{s['n_episodes']} | "
              f"{s['blocked_controllers']:2d}/10 | " + " / ".join(f"{s['switches_mean'][t]:.1f}" for t in TRAFFIC_SIGNAL_IDS))
    for arch, c in result["comparisons"].items():
        w = c["aligned_minus_phase3"]
        print(f"\n{arch}: corregido - Fase 3 = {w['d']:+.1f}, Welch p = {w['p']:.3g}, IC95 [{w['ci95'][0]:+.1f}, "
              f"{w['ci95'][1]:+.1f}]; criterios {c['criteria']} -> importante: {c['important']}; "
              f"plan_ppo_{arch} usa {result['plan_ppo'][arch]['uses']}")
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
