"""11.1 Instrument visual-search efficiency.

Same search episodes as search_time.py (every return to a target AOI),
scored as a path-efficiency ratio -- the same idea
controls_biomarkers/controls_efficiency.py uses for control inputs,
applied to the scanpath instead:

    direct_distance = straight-line angular distance from where the
                       search started to the target AOI's centroid
    path_length     = actual angular distance travelled hopping through
                       every intermediate fixation on the way
    efficiency       = direct_distance / path_length   (1.0 = went
                       straight there; lower = zig-zagged / rescanned)

Pick the target with --target-aoi (id or calibrated label); defaults to
the most-dwelled AOI (see incorrect_search_locations.py's default).

Limitation: "direct distance" is a straight line in yaw/pitch space, not
a claim about the physically shortest eye movement path (which is not
generally straight in yaw/pitch either) -- treat this as a relative
efficiency score across searches within the same sortie, not an absolute
optimal-path fraction.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, assign_aoi, fixations  # noqa: E402
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
    if target_aoi is None or not fx:
        return Result("instrument_search_efficiency", {"searches": 0}, [], "no fixations")

    centroid = np.mean([[f["yaw"], f["pitch"]] for f in fx if f["aoi"] == target_aoi], axis=0)

    episodes, i, n = [], 0, len(fx)
    while i < n:
        if fx[i]["aoi"] != target_aoi:
            i += 1
            continue
        j = i - 1
        while j >= 0 and fx[j]["aoi"] != target_aoi:
            j -= 1
        if j < 0:
            i += 1
            continue
        start_i = j + 1   # first fixation of the search (right after leaving target)
        path = sum(
            float(np.hypot(fx[k + 1]["yaw"] - fx[k]["yaw"], fx[k + 1]["pitch"] - fx[k]["pitch"]))
            for k in range(start_i, i)
        )
        direct = float(np.hypot(fx[start_i]["yaw"] - centroid[0], fx[start_i]["pitch"] - centroid[1]))
        if path > 1e-6:
            episodes.append({"t0": fx[start_i]["t0"], "t1": fx[i]["t0"],
                             "direct_deg": round(direct, 2), "path_deg": round(path, 2),
                             "efficiency": round(min(direct / path, 1.0), 3)})
        while i < n and fx[i]["aoi"] == target_aoi:
            i += 1

    effs = np.array([e["efficiency"] for e in episodes])
    metrics = {
        "target_aoi": int(target_aoi),
        "target_label": next((f["aoi_label"] for f in fx if f["aoi"] == target_aoi), None),
        "searches": len(episodes),
        "mean_efficiency": round(float(effs.mean()), 3) if len(effs) else None,
        "median_efficiency": round(float(np.median(effs)), 3) if len(effs) else None,
    }
    return Result("instrument_search_efficiency", metrics, episodes[:300],
                 limitations="relative path-efficiency score; see module docstring")


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
    print(f"instrument_search_efficiency: {result.metrics}")
    print(f"Wrote {out}")
