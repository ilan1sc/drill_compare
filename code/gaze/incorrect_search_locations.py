"""11.1 Incorrect search locations before reaching the required AOI.

Companion to search_time.py, same base.search_episodes() data, different
question: WHICH AOIs get visited as wrong stops on the way to a target,
and how often -- a table of candidate confusions (e.g. "checked AOI_2
before AOI_4 in 6 of 14 searches"), rather than just the count.

Pick the target with --target-aoi (id or calibrated label); without one,
defaults to the AOI with the most total dwell time (aoi_dwell_time.py's
top row) as a stand-in for "the most commonly required instrument".

Limitation: same AOI-is-unsupervised caveat as the rest of this package;
a "wrong stop" is only as meaningful as the target choice and the cluster
labels behind it.
"""

import sys
from pathlib import Path

import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, assign_aoi, fixations, search_episodes  # noqa: E402
from schema import GazeTrack  # noqa: E402


def _default_target(fx):
    dwell: dict[int, float] = {}
    for f in fx:
        dwell[f["aoi"]] = dwell.get(f["aoi"], 0.0) + f["dur"]
    return max(dwell, key=dwell.get) if dwell else None


def compute(track, target_aoi=None) -> Result:
    fx = [f for f in fixations(track) if f["kind"] == "fixation" and f["aoi"] is not None]
    if target_aoi is None:
        target_aoi = _default_target(fx)
    if target_aoi is None:
        return Result("incorrect_search_locations", {"searches": 0}, [], "no fixations")

    eps = search_episodes(fx, target_aoi)
    wrong = Counter(a for e in eps for a in e["aois_visited"])
    labels = {f["aoi"]: f["aoi_label"] for f in fx}
    counts = np.array([e["n_aois_visited"] for e in eps])

    table = [{"aoi": a, "aoi_label": labels.get(a), "times_visited_before_target": n,
             "share_of_searches": round(n / max(len(eps), 1), 3)}
            for a, n in wrong.most_common(15)]

    metrics = {
        "target_aoi": target_aoi, "target_label": labels.get(target_aoi),
        "searches": len(eps),
        "mean_incorrect_stops": round(float(counts.mean()), 2) if len(counts) else None,
        "zero_incorrect_rate": round(float(np.mean(counts == 0)), 3) if len(counts) else None,
        "most_common_wrong_aois": table,
    }
    return Result("incorrect_search_locations", metrics, eps[:300],
                 limitations="AOI = unsupervised gaze cluster; see module docstring")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gaze_json")
    ap.add_argument("--target-aoi", default=None)
    ap.add_argument("--aoi-k", type=int, default=6)
    args = ap.parse_args()

    track = GazeTrack.load(args.gaze_json)
    if not track.samples or track.samples[0].aoi is None:
        assign_aoi(track, k=args.aoi_k)
    target = int(args.target_aoi) if args.target_aoi and args.target_aoi.lstrip("-").isdigit() \
        else args.target_aoi
    result = compute(track, target_aoi=target)
    out = result.save(track.sortie, track.seat)
    print(f"incorrect_search_locations: {result.metrics}")
    print(f"Wrote {out}")
