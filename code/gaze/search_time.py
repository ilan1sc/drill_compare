"""11.1 Search Time -- when the cadet is looking inside the cockpit for a
specific instrument or information source, measure both:
    - Search duration [s]: elapsed time from the start of the visual
      search until the target AOI is reached.
    - AOIs visited before target [count]: number of distinct AOIs
      visited before reaching the target AOI.
Together these measure how directly and efficiently the cadet finds
required cockpit information. Interpretation remains exercise- and
context-dependent (per PRD 11.1 -- this module does not itself judge
"good" vs "bad" search, only measures it).

A "search" is every return to a target AOI (base.search_episodes): from
leaving whatever AOI was visited before, to landing back on the target.
Pick the target with --target-aoi (an AOI id from aoi_dwell_time.py, or
its calibrated label if aoi_labels.py has run); without one, this runs
for every AOI as its own target and reports each one's stats plus which
target had the fastest / most roundabout searches, so it's usable before
any calibration happens.

Limitation: AOI = unsupervised gaze cluster unless calibrated (see
schema.py); "the required instrument" is only as meaningful as whichever
AOI you pick as target.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, fixations, search_episodes, run_cli  # noqa: E402


def _aoi_for_arg(fx, arg):
    if arg is None:
        return None
    try:
        return int(arg)
    except ValueError:
        for f in fx:
            if f.get("aoi_label") == arg:
                return f["aoi"]
        return None


def compute(track, target_aoi=None) -> Result:
    fx = [f for f in fixations(track) if f["kind"] == "fixation" and f["aoi"] is not None]
    targets = [_aoi_for_arg(fx, target_aoi)] if target_aoi is not None \
        else sorted({f["aoi"] for f in fx})

    per_target, all_ep = {}, []
    for t in targets:
        if t is None:
            continue
        eps = search_episodes(fx, t)
        all_ep += eps
        durs = np.array([e["duration"] for e in eps])
        counts = np.array([e["n_aois_visited"] for e in eps])
        per_target[str(t)] = {
            "aoi_label": next((f["aoi_label"] for f in fx if f["aoi"] == t), None),
            "searches": len(eps),
            "mean_search_duration_s": round(float(durs.mean()), 2) if len(durs) else None,
            "median_search_duration_s": round(float(np.median(durs)), 2) if len(durs) else None,
            "mean_aois_before_target": round(float(counts.mean()), 2) if len(counts) else None,
            "direct_hit_rate": round(float(np.mean(counts == 0)), 3) if len(counts) else None,
        }

    metrics = {"targets_analyzed": len(per_target), "per_target": per_target}
    return Result("search_time", metrics, all_ep[:300],
                 limitations="target AOI must be chosen (--target-aoi); see module docstring")


if __name__ == "__main__":
    import argparse
    from schema import GazeTrack
    from base import assign_aoi

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gaze_json")
    ap.add_argument("--target-aoi", default=None, help="AOI id or calibrated label")
    ap.add_argument("--aoi-k", type=int, default=6)
    args = ap.parse_args()

    track = GazeTrack.load(args.gaze_json)
    if not track.samples or track.samples[0].aoi is None:
        assign_aoi(track, k=args.aoi_k)
    result = compute(track, target_aoi=args.target_aoi)
    out = result.save(track.sortie, track.seat)
    print(f"search_time: {result.metrics}")
    print(f"Wrote {out}")
