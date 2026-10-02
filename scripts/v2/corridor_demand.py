"""Demand candidates for the v2 four-intersection corridor, and the .rou.xml writer.

Every candidate tried during calibration stays here, so each iteration in
docs/v2/DISENO_RED_4_INTERSECCIONES.md can be regenerated exactly. The official
route file (environments/four-intersection-corridor/four-intersection-corridor.rou.xml)
is written from FINAL_CANDIDATE:

    python scripts/v2/corridor_demand.py                 # writes the official file
    python scripts/v2/corridor_demand.py --candidate it1 --out some/path.rou.xml

Demand model (veh/h):
- Arterial (East/West, green phase 1): ``arterial_west`` enters at the west end
  (left0, eastbound) and ``arterial_east`` at the east end (right0, westbound).
  At each of the 4 junctions a fraction ``arterial_turn_off`` of the entering
  flow turns off to the North and the same fraction to the South; the rest goes
  through to the opposite end.
- Cross streets (North/South, green phase 0): ``cross[J] = (north, south)``
  enters at junction J. A fraction ``cross_straight`` goes straight across;
  the rest turns onto the arterial, half towards each end of the corridor.
- ``arrivals``: "uniform" (``vehsPerHour``, evenly spaced, as in v1) or
  "poisson" (``period="exp(rate)"``, exponential headways).
- ``segments`` (optional): time-varying demand. A list of
  ``{"length": seconds, **overrides}`` played one after the other and repeated
  until the end of the route file; each override replaces that key of the base
  spec during its segment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.four_intersections import ROUTE_FILE, TRAFFIC_SIGNAL_IDS

# Junction index -> name of its fringe nodes (netgenerate: top0..top3, bottom0..bottom3).
JUNCTIONS = list(TRAFFIC_SIGNAL_IDS)

CANDIDATES: dict[str, dict] = {
    # Iteration 1: v1 style (evenly spaced arrivals), arterial heavier eastbound,
    # cross-street load different at every junction (C0 heavy, B0/D0 light).
    "it1": {
        "arrivals": "uniform",
        "arterial_west": 600,
        "arterial_east": 400,
        "arterial_turn_off": 0.0375,
        "cross": {"A0": (250, 150), "B0": (100, 60), "C0": (300, 200), "D0": (80, 120)},
        "cross_straight": 0.6,
    },
    # Iteration 2: it1 with Poisson arrivals. it1's evenly spaced flows sharing an
    # entry fire in synchronized bursts (the 8 turn-off flows of an arterial entry
    # have the same period), leaving 12-14 vehicles outside the network under every
    # policy, even "always arterial green": a backlog no controller causes or fixes.
    "it2": {
        "arrivals": "poisson",
        "arterial_west": 600,
        "arterial_east": 400,
        "arterial_turn_off": 0.0375,
        "cross": {"A0": (250, 150), "B0": (100, 60), "C0": (300, 200), "D0": (80, 120)},
        "cross_straight": 0.6,
    },
    # Iteration 3: it2 with a heavier arterial (750/450). In it1, switching as soon
    # as min_green allows was within noise of the best reference under the v1-style
    # reward: the load is low enough that the time lost to yellows costs nothing.
    "it3": {
        "arrivals": "poisson",
        "arterial_west": 750,
        "arterial_east": 450,
        "arterial_turn_off": 0.0375,
        "cross": {"A0": (250, 150), "B0": (100, 60), "C0": (300, 200), "D0": (80, 120)},
        "cross_straight": 0.6,
    },
    # Iteration 4: back to it2's arterial (750/450 left vehicles outside the network
    # even under the best reference), with strongly unequal cross streets: B0 and D0
    # almost empty, C0 heavy. A program shared by the four signals must be wrong
    # somewhere; switching as fast as possible wastes green on empty cross streets.
    "it4": {
        "arrivals": "poisson",
        "arterial_west": 600,
        "arterial_east": 400,
        "arterial_turn_off": 0.0375,
        "cross": {"A0": (200, 120), "B0": (40, 30), "C0": (350, 250), "D0": (40, 30)},
        "cross_straight": 0.6,
    },
    # Iteration 5: it4 plus 150 s pulses (period 300 s, one episode). First half:
    # eastbound arterial peak; second half: westbound arterial and a C0 cross-street
    # peak. The time-averaged demand is close to it4's, but no fixed program fits both halves.
    "it5": {
        "arrivals": "poisson",
        "arterial_west": 600,
        "arterial_east": 400,
        "arterial_turn_off": 0.0375,
        "cross": {"A0": (200, 120), "B0": (40, 30), "C0": (350, 250), "D0": (40, 30)},
        "cross_straight": 0.6,
        "segments": [
            {"length": 150, "arterial_west": 750, "arterial_east": 250,
             "cross": {"A0": (200, 120), "B0": (40, 30), "C0": (200, 150), "D0": (40, 30)}},
            {"length": 150, "arterial_west": 450, "arterial_east": 550,
             "cross": {"A0": (200, 120), "B0": (40, 30), "C0": (500, 350), "D0": (40, 30)}},
        ],
    },
}

FINAL_CANDIDATE = "it5"

VTYPE = ('<vType id="passenger" vClass="passenger" length="5" accel="2.6" decel="4.5" sigma="0.5" '
         'tau="1.0" maxSpeed="13.89" minGap="2.5" color="blue"/>')


def _arterial_edges(start: int, end: int) -> list[str]:
    """Arterial edges from junction index start to end (inclusive), either direction."""
    step = 1 if end > start else -1
    idx = list(range(start, end + step, step))
    return [f"{JUNCTIONS[a]}{JUNCTIONS[b]}" for a, b in zip(idx, idx[1:])]


def _routes(spec: dict) -> list[tuple[str, list[str], float]]:
    """(route_id, edges, veh/h) for every origin-destination pair with demand."""
    last = len(JUNCTIONS) - 1
    routes: list[tuple[str, list[str], float]] = []
    turn = spec["arterial_turn_off"]
    through = 1.0 - 2 * len(JUNCTIONS) * turn
    if through < 0:
        raise ValueError("arterial_turn_off too large: no arterial through traffic left")

    for origin, entry_edge, first, far_end, exit_edge in (
        ("west", "left0A0", 0, last, "D0right0"),
        ("east", "right0D0", last, 0, "A0left0"),
    ):
        total = spec[f"arterial_{origin}"]
        routes.append((f"{origin}_through", [entry_edge, *_arterial_edges(first, far_end), exit_edge],
                       total * through))
        for j, name in enumerate(JUNCTIONS):
            for side, fringe in (("north", "top"), ("south", "bottom")):
                routes.append((f"{origin}_to_{name}_{side}",
                               [entry_edge, *_arterial_edges(first, j), f"{name}{fringe}{j}"],
                               total * turn))

    for j, name in enumerate(JUNCTIONS):
        north, south = spec["cross"][name]
        for origin, total, entry_fringe, exit_fringe in (
            ("north", north, "top", "bottom"),
            ("south", south, "bottom", "top"),
        ):
            entry_edge = f"{entry_fringe}{j}{name}"
            straight = spec["cross_straight"]
            routes.append((f"{name}_{origin}_straight", [entry_edge, f"{name}{exit_fringe}{j}"], total * straight))
            turning = total * (1.0 - straight) / 2
            routes.append((f"{name}_{origin}_to_east",
                           [entry_edge, *_arterial_edges(j, last), "D0right0"], turning))
            routes.append((f"{name}_{origin}_to_west",
                           [entry_edge, *_arterial_edges(j, 0), "A0left0"], turning))
    return [r for r in routes if r[2] > 0]


def _segments(spec: dict, end: int) -> list[tuple[int, int, dict]]:
    """(begin, end, effective spec) for each demand segment up to ``end``."""
    if "segments" not in spec:
        return [(0, end, spec)]
    base = {k: v for k, v in spec.items() if k != "segments"}
    out, t, i = [], 0, 0
    while t < end:
        seg = spec["segments"][i % len(spec["segments"])]
        stop = min(end, t + seg["length"])
        out.append((t, stop, base | {k: v for k, v in seg.items() if k != "length"}))
        t, i = stop, i + 1
    return out


def build_routes_xml(spec: dict, name: str, end: int = 3600) -> str:
    if spec["arrivals"] not in ("uniform", "poisson"):
        raise ValueError(f"arrivals must be 'uniform' or 'poisson', got {spec['arrivals']!r}")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        f"    {VTYPE}",
        "",
        f"    <!-- Generated by scripts/v2/corridor_demand.py, candidate {name!r}. Do not edit by hand:",
        f"         change the candidate there and regenerate. Spec: {spec} -->",
        "",
    ]
    segments = _segments(spec, end)
    route_edges: dict[str, list[str]] = {}
    for _, _, seg_spec in segments:
        for route_id, edges, _ in _routes(seg_spec):
            route_edges.setdefault(route_id, edges)
    for route_id, edges in route_edges.items():
        lines.append(f'    <route id="{route_id}" edges="{" ".join(edges)}"/>')
    lines.append("")
    flows = []
    for n, (begin, stop, seg_spec) in enumerate(segments):
        suffix = "" if len(segments) == 1 else f"_s{n}"
        for route_id, _, vph in _routes(seg_spec):
            if spec["arrivals"] == "uniform":
                rate = f'vehsPerHour="{vph:.2f}"'
            else:
                rate = f'period="exp({vph / 3600:.6f})"'
            flows.append((begin, f'    <flow id="{route_id}_flow{suffix}" type="passenger" route="{route_id}" '
                                 f'begin="{begin}" end="{stop}" {rate} departLane="random" departSpeed="max"/>'))
    # SUMO requires route-file elements sorted by departure time.
    lines.extend(line for _, line in sorted(flows, key=lambda f: f[0]))
    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def write_routes(name: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_routes_xml(CANDIDATES[name], name), encoding="utf-8")
    return out


def total_demand(spec: dict, horizon: int = 300) -> float:
    """Mean veh/h entering the network over the first ``horizon`` seconds."""
    segs = _segments(spec, horizon)
    return sum((stop - begin) * sum(vph for _, _, vph in _routes(seg)) for begin, stop, seg in segs) / horizon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", default=FINAL_CANDIDATE, choices=sorted(CANDIDATES))
    parser.add_argument("--out", type=Path, default=ROUTE_FILE)
    args = parser.parse_args()
    path = write_routes(args.candidate, args.out)
    print(f"{args.candidate}: {total_demand(CANDIDATES[args.candidate]):.0f} veh/h -> {path}")


if __name__ == "__main__":
    main()
