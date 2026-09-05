"""11.1 AOI transitions and transition frequency.

The sequence of fixation AOIs (base.fixations, base.assign_aoi) gives a
transition graph: how often gaze moves from AOI i to AOI j. Reports the
transition matrix, overall transitions/min, and the most-worn transition
edges -- the closest thing to a scan PATTERN this pipeline can name
without instrument labels (see aoi_labels.py to attach real names).

Limitation: only records that a transition happened, not the physical
route the eyes took to get there.
"""

import sys
from pathlib import Path

from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, fixations, run_cli  # noqa: E402


def compute(track) -> Result:
    fx = [f for f in fixations(track) if f["kind"] == "fixation" and f["aoi"] is not None]
    seq = [f["aoi"] for f in fx]
    labels = {f["aoi"]: f["aoi_label"] for f in fx}

    edges = Counter(zip(seq, seq[1:]))
    edges = {k: v for k, v in edges.items() if k[0] != k[1]}   # a->a isn't a transition
    minutes = max(track.duration / 60.0, 1e-9)
    total = sum(edges.values())

    top = sorted(edges.items(), key=lambda kv: -kv[1])[:15]
    table = [{"from": a, "from_label": labels.get(a), "to": b, "to_label": labels.get(b),
             "count": n} for (a, b), n in top]

    metrics = {
        "transitions": total,
        "transitions_per_min": round(total / minutes, 2),
        "unique_edges": len(edges),
        "top_edges": table,
    }
    return Result("aoi_transitions", metrics, table,
                 limitations="AOI = unsupervised gaze cluster; see module docstring")


if __name__ == "__main__":
    run_cli("aoi_transitions", compute, needs_aoi=True)
