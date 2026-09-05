"""11.1 Gaze entropy / distribution of visual attention.

Shannon entropy of the AOI DWELL-TIME distribution (base.assign_aoi +
base.fixations), normalized to 0..1 by log(k). 0 = all attention on a
single AOI; 1 = attention spread perfectly evenly across every AOI
cluster. Distinct from scan_path_regularity.py's transition entropy:
this is about WHERE time was spent, that one is about the ORDER visits
happened in -- a pilot can have low path-transition entropy (a very
regular loop) while still spending very unevenly distributed time within
that loop (low dwell entropy), or vice versa.

Limitation: entropy is sensitive to the AOI count (k in --aoi-k); compare
entropy values computed with the same k, and prefer aoi_dwell_time.py's
per-AOI table when you need the actual concentration, not a single number.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, fixations, run_cli  # noqa: E402


def compute(track) -> Result:
    fx = [f for f in fixations(track) if f["kind"] == "fixation" and f["aoi"] is not None]
    if not fx:
        return Result("gaze_entropy", {"entropy_norm": None, "aois": 0}, [], "no fixations")

    by_aoi: dict[int, float] = {}
    for f in fx:
        by_aoi[f["aoi"]] = by_aoi.get(f["aoi"], 0.0) + f["dur"]
    total = sum(by_aoi.values())
    p = np.array([v / total for v in by_aoi.values()])
    entropy = float(-(p * np.log(p)).sum())
    norm = entropy / np.log(len(p)) if len(p) > 1 else 0.0

    metrics = {
        "aois": len(by_aoi), "entropy_nats": round(entropy, 3), "entropy_norm": round(norm, 3),
        "dominant_aoi_share": round(float(p.max()), 3),
    }
    return Result("gaze_entropy", metrics, [],
                 limitations="sensitive to --aoi-k; see module docstring")


if __name__ == "__main__":
    run_cli("gaze_entropy", compute, needs_aoi=True)
