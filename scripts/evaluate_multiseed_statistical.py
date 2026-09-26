"""Multi-seed evaluation on real SUMO, with the statistics of the central result.

Evaluates several policies -- each method with N training seeds -- on the same
real SUMO scenarios, writes every episode to a versionable JSON + CSV, and prints
the comparison table and tests used in PROJECT_STATUS.md:

- Welch t-test at EPISODE level (all episodes of a method pooled). Declared as
  pseudoreplication in the output: N episodes of k trained policies are not N
  independent observations of the training method.
- Welch t-test at SEED level (one mean per training seed): the test that says
  something about the method, with n = number of seeds (low power).
- PAIRED t-test by scenario: every policy runs the same scenario seeds, so the
  per-scenario difference (averaged over each method's seeds) removes the
  variance shared by the scenario. It generalizes over scenarios for THESE
  trained policies, not over training seeds.
- Mann-Whitney U at episode level (normal approximation, tie-corrected), kept
  for continuity with the tables already published; same pseudoreplication caveat.

scipy is not a project dependency, so the t and normal distributions are
computed here (regularized incomplete beta, math.erfc).

Policy specification, one --policy per method:
    NAME=dream:CKPT[,CKPT...]    PPO trained in the Dream Environment (sees z via EncodedTrafficEnvironment)
    NAME=direct:CKPT[,CKPT...]   PPO trained on real SUMO (raw state, its own VecNormalize statistics)
    NAME=fixed                   fixed-time control used throughout the project (action 1 every 5th step)
    NAME=rule                    "always request the opposite phase" (action = 1 - green_phase)
CKPT is a best_model.zip or a folder containing one; each CKPT is one training seed.

Example (the protocol used throughout the project: seed bases 3000 and 5000, 15 episodes each):
    python scripts/evaluate_multiseed_statistical.py \\
        --policy sueno=dream:models/checkpoints/controller/best_model_seed0_worse.zip,models/checkpoints/controller/best_model.zip,models/checkpoints/controller/best_model_seed2.zip \\
        --policy directo_10k=direct:models/checkpoints/controller_direct_fixed/seed0,models/checkpoints/controller_direct_fixed/seed1,models/checkpoints/controller_direct_fixed/seed2 \\
        --policy tiempo_fijo=fixed --policy regla=rule \\
        --compare sueno:directo_10k --output docs/results/multiseed_official_scenarios
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CATASTROPHIC_THRESHOLD = -600.0
FIXED_TIME_SWITCH_EVERY = 5


# ---------------------------------------------------------------------------
# Statistics (no scipy)
# ---------------------------------------------------------------------------

def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), continued fraction (Numerical Recipes)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def continued_fraction(a, b, x):
        tiny = 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, 500):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c if abs(c) > tiny else 1.0 + aa / tiny
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c if abs(c) > tiny else 1.0 + aa / tiny
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-14:
                break
        return h

    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * continued_fraction(a, b, x) / a
    return 1.0 - bt * continued_fraction(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    if not math.isfinite(t):
        return 0.0
    return _incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def welch(a, b) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "df": None, "p": None, "note": "needs >= 2 values per group"}
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return {"t": None, "df": None, "p": None, "note": "zero variance in both groups"}
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return {"t": t, "df": df, "p": t_two_sided_p(t, df)}


def paired_t(a, b) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return {"t": None, "df": None, "p": None, "mean_diff": float(d.mean()) if len(d) else None}
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    return {"t": t, "df": len(d) - 1, "p": t_two_sided_p(t, len(d) - 1), "mean_diff": float(d.mean())}


def mann_whitney(a, b) -> dict:
    """Two-sided Mann-Whitney U, normal approximation with tie correction."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    values = np.concatenate([a, b])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    i = 0
    tie_term = 0.0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        tie_term += (j - i + 1) ** 3 - (j - i + 1)
        i = j + 1
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u1 - n1 * n2 / 2.0) / sigma if sigma > 0 else 0.0
    return {"U": u1, "z": z, "p": math.erfc(abs(z) / math.sqrt(2.0))}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    name: str
    kind: str
    checkpoints: list[Path] = field(default_factory=list)


def parse_policy(spec: str) -> Policy:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--policy needs NAME=KIND[:CKPT,...], got {spec!r}")
    name, rest = spec.split("=", 1)
    kind, _, paths = rest.partition(":")
    if kind not in {"dream", "direct", "fixed", "rule"}:
        raise argparse.ArgumentTypeError(f"unknown policy kind {kind!r} in {spec!r}")
    checkpoints = []
    for raw in filter(None, paths.split(",")):
        path = Path(raw)
        if path.is_dir():
            path = path / "best_model.zip"
        if not path.exists():
            raise argparse.ArgumentTypeError(f"checkpoint not found: {path}")
        checkpoints.append(path)
    if kind in {"dream", "direct"} and not checkpoints:
        raise argparse.ArgumentTypeError(f"{kind} policy {name!r} needs at least one checkpoint")
    if kind in {"fixed", "rule"} and checkpoints:
        raise argparse.ArgumentTypeError(f"{kind} policy {name!r} takes no checkpoint")
    return Policy(name, kind, checkpoints)


def _green_phase(raw_env) -> int:
    sumo_env = raw_env.env
    return int(sumo_env.traffic_signals[sumo_env.ts_ids[0]].green_phase)


def build_action_functions(policy: Policy, encoded_env) -> list[tuple[str, callable]]:
    """One (label, fn(raw_state, step) -> action) per training seed of the policy."""
    from stable_baselines3 import PPO

    from training.train_controller import load_obs_normalizer

    raw_env = encoded_env._env
    if policy.kind == "fixed":
        return [("deterministic", lambda state, step: 1 if step % FIXED_TIME_SWITCH_EVERY == 0 else 0)]
    if policy.kind == "rule":
        return [("deterministic", lambda state, step: 1 - _green_phase(raw_env))]

    functions = []
    for checkpoint in policy.checkpoints:
        model = PPO.load(checkpoint)
        if policy.kind == "dream":
            fn = (lambda m: lambda state, step: int(m.predict(encoded_env._encode(state), deterministic=True)[0]))(model)
        else:
            normalize = load_obs_normalizer(checkpoint, raw_env)
            fn = (lambda m, n: lambda state, step: int(m.predict(n(state), deterministic=True)[0]))(model, normalize)
        functions.append((str(checkpoint.as_posix()), fn))
    return functions


def run_episode(raw_env, action_fn, scenario_seed: int) -> dict:
    state, _ = raw_env.reset(seed=scenario_seed)
    total_reward = total_waiting = total_queue = 0.0
    switches = step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        state, reward, terminated, truncated, info = raw_env.step(action_fn(state, step))
        total_reward += reward
        total_waiting += info["waiting_total"]
        total_queue += info["queue_total"]
        switches += int(info["phase_switched"])
        step += 1
    return {
        "reward": float(total_reward),
        "waiting_mean": total_waiting / step,
        "queue_mean": total_queue / step,
        "phase_switches": switches,
        "steps": step,
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize(episodes: list[dict], policy_names: list[str], scenarios: list[int]) -> dict:
    summary = {}
    for name in policy_names:
        rows = [e for e in episodes if e["policy"] == name]
        rewards = np.array([e["reward"] for e in rows])
        seeds = sorted({e["seed_label"] for e in rows})
        seed_means = [float(np.mean([e["reward"] for e in rows if e["seed_label"] == s])) for s in seeds]
        summary[name] = {
            "n_episodes": len(rows),
            "n_seeds": len(seeds),
            "mean": float(rewards.mean()),
            "median": float(np.median(rewards)),
            "std": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
            "stderr": float(rewards.std(ddof=1) / math.sqrt(len(rewards))) if len(rewards) > 1 else 0.0,
            "seed_labels": seeds,
            "seed_means": seed_means,
            "std_of_seed_means": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None,
            "catastrophic": int((rewards < CATASTROPHIC_THRESHOLD).sum()),
            "worst": float(rewards.min()),
            "waiting_mean": float(np.mean([e["waiting_mean"] for e in rows])),
            "queue_mean": float(np.mean([e["queue_mean"] for e in rows])),
            "per_scenario_mean": [
                float(np.mean([e["reward"] for e in rows if e["scenario_seed"] == sc])) for sc in scenarios
            ],
        }
    return summary


def compare(summary: dict, episodes: list[dict], a: str, b: str, scenarios: list[int]) -> dict:
    ra = [e["reward"] for e in episodes if e["policy"] == a]
    rb = [e["reward"] for e in episodes if e["policy"] == b]
    sa, sb = summary[a], summary[b]
    # Per scenario, every seed of A against every seed of B (same traffic).
    wins = total = 0
    for sc in scenarios:
        va = [e["reward"] for e in episodes if e["policy"] == a and e["scenario_seed"] == sc]
        vb = [e["reward"] for e in episodes if e["policy"] == b and e["scenario_seed"] == sc]
        for x, y in itertools.product(va, vb):
            wins += int(x > y)
            total += 1
    return {
        "a": a,
        "b": b,
        "mean_diff": sa["mean"] - sb["mean"],
        "episode_level_welch": welch(ra, rb),
        "episode_level_mann_whitney": mann_whitney(ra, rb),
        "seed_level_welch": welch(sa["seed_means"], sb["seed_means"]),
        "paired_by_scenario": paired_t(sa["per_scenario_mean"], sb["per_scenario_mean"]),
        "a_beats_b_same_scenario": {"wins": wins, "comparisons": total},
    }


def _fmt_test(result: dict) -> str:
    if result.get("p") is None:
        return "n/a (" + result.get("note", "insufficient data") + ")"
    df = f", gl={result['df']:.1f}" if result.get("df") is not None else ""
    stat = f"t={result['t']:.2f}" if "t" in result else f"z={result['z']:.2f}"
    return f"{stat}{df}, p={result['p']:.3g}"


def print_report(summary: dict, comparisons: list[dict], scenarios: list[int]) -> None:
    print()
    print(f"Escenarios: {len(scenarios)} semillas de SUMO ({scenarios[0]}..{scenarios[-1]})")
    header = (f"{'Politica':16s} | {'sem.':>4s} | {'ep.':>4s} | {'media':>9s} | {'mediana':>9s} | {'desv':>8s} | "
              f"{'desv medias/sem.':>16s} | {'<-600':>5s} | {'peor':>9s} | {'espera':>6s} | {'cola':>5s} | medias por semilla")
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        std_seeds = f"{s['std_of_seed_means']:.1f}" if s["std_of_seed_means"] is not None else "-"
        print(f"{name:16s} | {s['n_seeds']:4d} | {s['n_episodes']:4d} | {s['mean']:9.2f} | {s['median']:9.2f} | "
              f"{s['std']:8.2f} | {std_seeds:>16s} | {s['catastrophic']:5d} | {s['worst']:9.1f} | "
              f"{s['waiting_mean']:6.2f} | {s['queue_mean']:5.2f} | {[round(m, 2) for m in s['seed_means']]}")
    for c in comparisons:
        sa, sb = summary[c["a"]], summary[c["b"]]
        w = c["a_beats_b_same_scenario"]
        print()
        print(f"{c['a']} vs {c['b']}: diferencia de medias {c['mean_diff']:+.2f}")
        print(f"  Nivel semilla (Welch, {sa['n_seeds']} vs {sb['n_seeds']} medias): {_fmt_test(c['seed_level_welch'])}"
              "  <- el unico que prueba algo sobre el METODO; potencia baja con pocas semillas")
        print(f"  Pareado por escenario ({len(scenarios)} escenarios, media de las semillas de cada metodo): "
              f"{_fmt_test(c['paired_by_scenario'])}  <- generaliza sobre escenarios para ESTAS politicas, "
              "no sobre semillas de entrenamiento")
        print(f"  Nivel episodio (Welch, {sa['n_episodes']} vs {sb['n_episodes']}): {_fmt_test(c['episode_level_welch'])}"
              f"  <- PSEUDORREPLICACION: {sa['n_episodes']} episodios de solo {sa['n_seeds']} politica(s)")
        print(f"  Nivel episodio (Mann-Whitney): {_fmt_test(c['episode_level_mann_whitney'])}  <- misma salvedad")
        print(f"  {c['a']} gana a {c['b']} en {w['wins']}/{w['comparisons']} comparaciones sobre el mismo escenario")


def save_results(output: Path, args_record: dict, episodes, summary, comparisons) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"arguments": args_record, "summary": summary, "comparisons": comparisons, "episodes": episodes}
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = ["policy", "kind", "seed_label", "scenario_seed", "reward", "waiting_mean", "queue_mean",
              "phase_switches", "steps"]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for e in episodes:
            writer.writerow({k: e[k] for k in fields})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", action="append", type=parse_policy, required=True)
    parser.add_argument("--seed-bases", type=int, nargs="+", default=[3000, 5000])
    parser.add_argument("--episodes-per-base", type=int, default=15)
    parser.add_argument("--compare", action="append", default=[], help="A:B, repeatable (default: every pair)")
    parser.add_argument("--output", type=Path, required=True, help="output path without extension (.json and .csv)")
    args = parser.parse_args(argv)

    names = [p.name for p in args.policy]
    if len(set(names)) != len(names):
        parser.error("policy names must be unique")
    pairs = [tuple(c.split(":", 1)) for c in args.compare] or list(itertools.combinations(names, 2))
    for a, b in pairs:
        if a not in names or b not in names:
            parser.error(f"--compare {a}:{b} names an unknown policy")

    scenarios = [base + i for base in args.seed_bases for i in range(args.episodes_per_base)]

    from environments.encoded_traffic_environment import EncodedTrafficEnvironment

    # One SUMO simulation for everything: every episode starts from its own reset(seed).
    encoded_env = EncodedTrafficEnvironment()
    raw_env = encoded_env._env
    episodes = []
    try:
        for policy in args.policy:
            for seed_label, action_fn in build_action_functions(policy, encoded_env):
                for scenario in scenarios:
                    record = run_episode(raw_env, action_fn, scenario)
                    record.update({"policy": policy.name, "kind": policy.kind, "seed_label": seed_label,
                                   "scenario_seed": scenario})
                    episodes.append(record)
                seed_rewards = [e["reward"] for e in episodes if e["policy"] == policy.name and e["seed_label"] == seed_label]
                print(f"{policy.name:16s} {seed_label}: media {np.mean(seed_rewards):.2f}", flush=True)
    finally:
        encoded_env.close()

    summary = summarize(episodes, names, scenarios)
    comparisons = [compare(summary, episodes, a, b, scenarios) for a, b in pairs]
    print_report(summary, comparisons, scenarios)
    args_record = {
        "policies": [{"name": p.name, "kind": p.kind, "checkpoints": [c.as_posix() for c in p.checkpoints]}
                     for p in args.policy],
        "seed_bases": args.seed_bases,
        "episodes_per_base": args.episodes_per_base,
        "scenarios": scenarios,
    }
    save_results(args.output, args_record, episodes, summary, comparisons)
    print(f"\nResultados por episodio: {args.output.with_suffix('.json')} y {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
