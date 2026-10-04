"""Evaluate v2 controllers and references on real SUMO, per episode and per intersection
(docs/v2/ADDENDUM_CONTROL.md).

Policy specification, one --policy per method:
    NAME=dream:CKPT[,CKPT...]    PPO trained in CorridorDreamEnvironment (sees the scaler.pkl-normalized state)
    NAME=direct:CKPT[,CKPT...]   PPO trained on real SUMO (raw state, its own VecNormalize statistics)
    NAME=ref:POLICY              one of the five non-learned references (scripts/v2/corridor_policies.py)
CKPT is a best_model.zip or a folder holding one; each CKPT is one training seed.

Scenarios: --split validation|test|ood (all its seeds, or --seeds, a subset of them). The train
seeds 20000-20111 are refused, and test / ood need --confirm-held-out (addendum, section 2).

Writes every episode (total return, return per intersection A0..D0, waiting, queue, arrivals,
real phase switches per signal) to <output>.json and .csv, and prints per-policy summaries.
--compare A:B[:METRIC] adds the addendum's tests (section 5) for METRIC = total (default) or a
signal id: seed-level Welch (one-sample t against a deterministic reference), paired t and paired
Wilcoxon by scenario, with the CI at --level.

--reference-agreement: at every step of every episode, also asks the five references which
keep/switch they WOULD choose for each signal from that same state, without executing it, and
records the fraction of steps each one agrees with the evaluated policy (agree_<ref>_<signal>).
It only reads the simulation, so the trajectory is the evaluated policy's own
(tests/test_v2_control.py checks that returns do not change with it on).

    python scripts/v2/evaluate_control_v2.py --split validation --policy fijo_2_3=ref:fijo_2_3 --output <path>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "scripts" / "v2") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts" / "v2"))

from environments.four_intersections import TRAFFIC_SIGNAL_IDS  # noqa: E402

SPLIT_SEEDS = {
    "validation": range(21000, 21024),
    "test": range(22000, 22024),
    "ood": range(23000, 23030),
}
HELD_OUT_SPLITS = ("test", "ood")
TRAIN_SPLIT_SEEDS = range(20000, 20112)
METRICS = ("total", *TRAFFIC_SIGNAL_IDS)
# docs/v2/ADDENDUM_CONTROL.md, section 4.3: 1.5 x the worst fijo_2_3 return on the 24 validation
# seeds, rounded down to the hundred: worst -2,354.0 (fijo_2_3, seed 21000)
# -> 1.5 x -2,354.0 = -3,531 -> -3,600. Fixed on 4 October 2026 from references_validation.json.
CATASTROPHIC_THRESHOLD: float | None = -3600.0


@dataclass
class Policy:
    name: str
    kind: str
    checkpoints: list[Path] = field(default_factory=list)
    reference: str | None = None


def parse_policy(spec: str) -> Policy:
    from scripts.v2.corridor_policies import REFERENCE_POLICIES

    if "=" not in spec or ":" not in spec.split("=", 1)[1]:
        raise argparse.ArgumentTypeError(f"--policy needs NAME=KIND:ARG, got {spec!r}")
    name, rest = spec.split("=", 1)
    kind, arg = rest.split(":", 1)
    if kind == "ref":
        if arg not in REFERENCE_POLICIES:
            raise argparse.ArgumentTypeError(f"unknown reference {arg!r}; expected one of {REFERENCE_POLICIES}")
        return Policy(name, kind, reference=arg)
    if kind not in {"dream", "direct"}:
        raise argparse.ArgumentTypeError(f"unknown policy kind {kind!r} in {spec!r}")
    checkpoints = []
    for raw in filter(None, arg.split(",")):
        path = Path(raw)
        path = path / "best_model.zip" if path.is_dir() else path
        if not path.exists():
            raise argparse.ArgumentTypeError(f"checkpoint not found: {path}")
        checkpoints.append(path)
    if not checkpoints:
        raise argparse.ArgumentTypeError(f"{kind} policy {name!r} needs at least one checkpoint")
    return Policy(name, kind, checkpoints)


def scenario_seeds(split: str, seeds: list[int] | None, confirm_held_out: bool) -> list[int]:
    if split in HELD_OUT_SPLITS and not confirm_held_out:
        raise SystemExit(f"--split {split} is held out (ADDENDUM_CONTROL.md, section 2): pass --confirm-held-out "
                         "only for the pre-registered final evaluation.")
    allowed = list(SPLIT_SEEDS[split])
    chosen = allowed if seeds is None else list(seeds)
    bad = [s for s in chosen if s in TRAIN_SPLIT_SEEDS or s not in allowed]
    if bad:
        raise SystemExit(f"seeds {bad} are not {split} seeds (train seeds 20000-20111 are never evaluated on)")
    return chosen


def build_action_functions(policy: Policy, scaled_env) -> list[tuple[str, callable]]:
    """One (label, fn(raw_state, step) -> joint action) per training seed of the policy."""
    raw_env = scaled_env.env
    if policy.kind == "ref":
        from scripts.v2.corridor_policies import make_reference_policy

        act = make_reference_policy(policy.reference)
        return [("deterministic", lambda state, step: act(raw_env, step))]

    from stable_baselines3 import PPO

    from training.train_controller import load_obs_normalizer

    functions = []
    for checkpoint in policy.checkpoints:
        model = PPO.load(checkpoint, device="cpu")
        if policy.kind == "dream":
            obs = scaled_env.observation  # the same scaler.pkl the dream was built on
        else:
            obs = load_obs_normalizer(checkpoint, raw_env)
        functions.append((checkpoint.as_posix(),
                          (lambda m, o: lambda state, step: np.asarray(m.predict(o(state), deterministic=True)[0]))(model, obs)))
    return functions


def run_episode(raw_env, action_fn, scenario_seed: int, shadow_references: dict | None = None) -> dict:
    t0 = time.perf_counter()
    shadow_references = shadow_references or {}
    agree = {name: np.zeros(len(TRAFFIC_SIGNAL_IDS)) for name in shadow_references}
    state, info = raw_env.reset(seed=scenario_seed)
    per_signal = dict.fromkeys(TRAFFIC_SIGNAL_IDS, 0.0)
    switches = np.zeros(len(TRAFFIC_SIGNAL_IDS))
    total = waiting = queue = arrivals = 0.0
    step, done = 0, False
    while not done:
        action = np.asarray(action_fn(state, step), dtype=np.int64)
        for name, reference in shadow_references.items():  # same state, before the step; never executed
            agree[name] += np.asarray(reference(raw_env, step)) == action
        state, reward, terminated, truncated, step_info = raw_env.step(action)
        total += reward
        for ts, r in step_info["reward_per_signal"].items():
            per_signal[ts] += r
        switches += step_info["phase_switched"]
        waiting += step_info["waiting_total"]
        queue += step_info["queue_total"]
        arrivals += step_info["arrivals_total"]
        step, done = step + 1, terminated or truncated
    record = {"reward": float(total), **{f"reward_{ts}": float(v) for ts, v in per_signal.items()},
              "waiting_mean": waiting / step, "queue_mean": queue / step, "arrivals": arrivals,
              **{f"switches_{ts}": int(s) for ts, s in zip(TRAFFIC_SIGNAL_IDS, switches)},
              "pulse_offset": int(info["pulse_offset"]), "steps": step, "seconds": time.perf_counter() - t0}
    for name, counts in agree.items():
        record |= {f"agree_{name}_{ts}": float(c / step) for ts, c in zip(TRAFFIC_SIGNAL_IDS, counts)}
    return record


def metric_key(metric: str) -> str:
    return "reward" if metric == "total" else f"reward_{metric}"


def summarize(episodes: list[dict], names: list[str], scenarios: list[int], threshold: float | None) -> dict:
    summary = {}
    for name in names:
        rows = [e for e in episodes if e["policy"] == name]
        labels = sorted({e["seed_label"] for e in rows})
        entry = {"n_seeds": len(labels), "n_episodes": len(rows), "seed_labels": labels}
        for metric in METRICS:
            key = metric_key(metric)
            values = np.array([e[key] for e in rows])
            entry[metric] = {
                "mean": float(values.mean()),
                "seed_means": [float(np.mean([e[key] for e in rows if e["seed_label"] == s])) for s in labels],
                "per_scenario_mean": [float(np.mean([e[key] for e in rows if e["scenario_seed"] == sc]))
                                      for sc in scenarios],
                "worst": float(values.min()),
            }
        totals = np.array([e["reward"] for e in rows])
        entry["catastrophic"] = None if threshold is None else int((totals < threshold).sum())
        for extra in ("waiting_mean", "queue_mean", "arrivals"):
            entry[extra] = float(np.mean([e[extra] for e in rows]))
        entry["switches_mean"] = {ts: float(np.mean([e[f"switches_{ts}"] for e in rows])) for ts in TRAFFIC_SIGNAL_IDS}
        agree_keys = sorted(k for k in rows[0] if k.startswith("agree_"))
        if agree_keys:
            entry["agreement_mean"] = {k[len("agree_"):]: float(np.mean([e[k] for e in rows])) for k in agree_keys}
        summary[name] = entry
    return summary


def compare(summary: dict, a: str, b: str, metric: str, level: float) -> dict:
    """The addendum's three tests of mean(a) - mean(b) on ``metric`` (section 5)."""
    import analyze_lstm_vs_transformer as stats  # the project's t distribution and Welch (no scipy)
    from scripts.evaluate_multiseed_statistical import paired_t
    from scripts.v2.validate_corridor_demand import wilcoxon

    sa, sb = np.array(summary[a][metric]["seed_means"]), np.array(summary[b][metric]["seed_means"])
    pa, pb = np.array(summary[a][metric]["per_scenario_mean"]), np.array(summary[b][metric]["per_scenario_mean"])
    if len(sa) >= 2 and len(sb) >= 2:
        seed_test = {"test": "welch", **stats.welch(sa, sb, level)}
    elif len(sa) >= 2 or len(sb) >= 2:
        # One side deterministic (a reference): one-sample t of the learned side's seed means minus
        # the reference's mean on the same scenarios, signed as a - b.
        learned, ref, sign = (sa, sb, 1.0) if len(sa) >= 2 else (sb, sa, -1.0)
        diffs = sign * (learned - ref.mean())
        se = diffs.std(ddof=1) / math.sqrt(len(diffs))
        t = diffs.mean() / se
        half = stats.t_quantile(round(1.0 - (1.0 - level) / 2.0, 12), len(diffs) - 1) * se
        seed_test = {"test": "one_sample_t", "d": float(diffs.mean()), "t": float(t), "df": len(diffs) - 1,
                     "p": stats.t_two_sided_p(t, len(diffs) - 1), "ci95": [diffs.mean() - half, diffs.mean() + half]}
    else:
        seed_test = {"test": None, "note": "both sides deterministic"}
    if "ci95" in seed_test:
        seed_test["ci"] = [float(v) for v in seed_test.pop("ci95")]
    return {"a": a, "b": b, "metric": metric, "level": level,
            "mean_diff": float(summary[a][metric]["mean"] - summary[b][metric]["mean"]),
            "seed_level": seed_test, "paired_t": paired_t(pa, pb), "wilcoxon": wilcoxon(pa, pb),
            "a_wins_scenarios": int((pa > pb).sum()), "n_scenarios": len(pa)}


def print_report(summary: dict, comparisons: list[dict], scenarios: list[int]) -> None:
    print(f"\nEscenarios: {len(scenarios)} ({scenarios[0]}..{scenarios[-1]})")
    header = (f"{'Politica':22s} | sem. | {'total':>9s} | " + " | ".join(f"{ts:>8s}" for ts in TRAFFIC_SIGNAL_IDS)
              + f" | {'peor':>8s} | catastr. | medias por semilla (total)")
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        cat = "-" if s["catastrophic"] is None else str(s["catastrophic"])
        print(f"{name:22s} | {s['n_seeds']:4d} | {s['total']['mean']:9.1f} | "
              + " | ".join(f"{s[ts]['mean']:8.1f}" for ts in TRAFFIC_SIGNAL_IDS)
              + f" | {s['total']['worst']:8.1f} | {cat:>8s} | {[round(m, 1) for m in s['total']['seed_means']]}")
    for c in comparisons:
        sl = c["seed_level"]
        seed_txt = (f"{sl['test']}: d={sl['d']:+.1f}, p={sl['p']:.3g}, IC{c['level']:.2%}=[{sl['ci'][0]:+.1f}, {sl['ci'][1]:+.1f}]"
                    if sl.get("test") else sl["note"])
        pt = c["paired_t"]
        pt_txt = f"p={pt['p']:.3g}" if pt.get("p") is not None else "n/a"
        print(f"\n{c['a']} - {c['b']} ({c['metric']}): {c['mean_diff']:+.1f}; nivel semilla {seed_txt}; "
              f"t pareada {pt_txt}; Wilcoxon p={c['wilcoxon']['p']:.3g}; gana {c['a_wins_scenarios']}/{c['n_scenarios']}")


def save_results(output: Path, record: dict, episodes: list[dict], summary: dict, comparisons: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps({"arguments": record, "summary": summary,
                                                       "comparisons": comparisons, "episodes": episodes},
                                                      indent=1, default=float), encoding="utf-8")
    fields = ["policy", "kind", "seed_label", "scenario_seed", "reward", *[f"reward_{ts}" for ts in TRAFFIC_SIGNAL_IDS],
              "waiting_mean", "queue_mean", "arrivals", *[f"switches_{ts}" for ts in TRAFFIC_SIGNAL_IDS],
              "pulse_offset", "steps", "seconds"]
    fields += sorted({k for e in episodes for k in e if k.startswith("agree_")})
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for e in episodes:
            writer.writerow({k: e.get(k) for k in fields})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", action="append", type=parse_policy, required=True)
    parser.add_argument("--split", choices=sorted(SPLIT_SEEDS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="subset of the split's seeds")
    parser.add_argument("--confirm-held-out", action="store_true", help="required for --split test / ood")
    parser.add_argument("--compare", action="append", default=[], help="A:B[:METRIC], METRIC in total, A0..D0")
    parser.add_argument("--level", type=float, default=0.95, help="CI level (the addendum's corrected one: 1 - 0.05/6)")
    parser.add_argument("--reference-agreement", action="store_true",
                        help="record, per step, whether each reference would have chosen the same action")
    parser.add_argument("--catastrophic-threshold", type=float, default=CATASTROPHIC_THRESHOLD)
    parser.add_argument("--output", type=Path, required=True, help="output path without extension")
    args = parser.parse_args(argv)

    names = [p.name for p in args.policy]
    if len(set(names)) != len(names):
        parser.error("policy names must be unique")
    pairs = []
    for spec in args.compare:
        a, b, *rest = spec.split(":")
        metric = rest[0] if rest else "total"
        if a not in names or b not in names or metric not in METRICS:
            parser.error(f"--compare {spec}: unknown policy or metric")
        pairs.append((a, b, metric))
    scenarios = scenario_seeds(args.split, args.seeds, args.confirm_held_out)

    from environments.scaled_corridor_environment import ScaledCorridorEnvironment

    scaled_env = ScaledCorridorEnvironment()  # one simulation; every episode starts from its own reset(seed)
    raw_env = scaled_env.env
    shadow = {}
    if args.reference_agreement:
        from scripts.v2.corridor_policies import REFERENCE_POLICIES, make_reference_policy

        shadow = {name: make_reference_policy(name) for name in REFERENCE_POLICIES}
    episodes = []
    try:
        for policy in args.policy:
            for label, action_fn in build_action_functions(policy, scaled_env):
                for scenario in scenarios:
                    record = run_episode(raw_env, action_fn, scenario, shadow)
                    record.update({"policy": policy.name, "kind": policy.kind, "seed_label": label,
                                   "scenario_seed": scenario})
                    episodes.append(record)
                rewards = [e["reward"] for e in episodes if e["policy"] == policy.name and e["seed_label"] == label]
                print(f"{policy.name:22s} {label}: media {np.mean(rewards):.1f}", flush=True)
    finally:
        scaled_env.close()

    summary = summarize(episodes, names, scenarios, args.catastrophic_threshold)
    comparisons = [compare(summary, a, b, metric, args.level) for a, b, metric in pairs]
    print_report(summary, comparisons, scenarios)
    record = {"split": args.split, "scenarios": scenarios, "level": args.level,
              "reference_agreement": args.reference_agreement,
              "catastrophic_threshold": args.catastrophic_threshold,
              "policies": [{"name": p.name, "kind": p.kind, "reference": p.reference,
                            "checkpoints": [c.as_posix() for c in p.checkpoints]} for p in args.policy]}
    save_results(args.output, record, episodes, summary, comparisons)
    print(f"\n-> {args.output.with_suffix('.json')} y .csv")


if __name__ == "__main__":
    main()
