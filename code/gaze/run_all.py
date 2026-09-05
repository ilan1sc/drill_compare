"""Run the 13 single-track gaze biomarkers (PRD 11.1) over one seat's
gaze track and write a combined summary. (search_time.py,
instrument_search_efficiency.py and incorrect_search_locations.py need a
--target-aoi choice and time_to_first_fixation.py an optional
--events-csv, so they're run here with defaults -- rerun them directly
for a specific target/events.) scan_strategy_change.py is cross-sortie
and run separately once >=2 sorties exist for a seat.

    python3 code/gaze/run_all.py "/home/ilan/Documents/flight school/data/s1 c2.mp4"
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import GAZE_DIR, assign_aoi  # noqa: E402
import extract_gaze  # noqa: E402

SIMPLE_MODULES = [
    "gaze_direction", "fixation_duration_frequency", "saccade_frequency_velocity",
    "aoi_dwell_time", "aoi_transitions", "scan_path_regularity", "gaze_entropy",
    "time_inside_vs_outside_panel", "head_eye_stability",
]
TARGETED_MODULES = ["search_time", "instrument_search_efficiency", "incorrect_search_locations",
                    "time_to_first_fixation"]


def run_one_track(track, aoi_k: int = 6) -> dict:
    if not track.samples:
        return {"seat": track.seat, "samples": 0, "biomarkers": {}}
    assign_aoi(track, k=aoi_k)   # compute once, share across every module below

    summary = {"seat": track.seat, "samples": len(track.samples), "biomarkers": {}}
    for name in SIMPLE_MODULES:
        mod = importlib.import_module(name)
        result = mod.compute(track)
        result.save(track.sortie, track.seat)
        summary["biomarkers"][name] = result.metrics
        print(f"  [{track.seat}] {name}: {json.dumps(result.metrics, ensure_ascii=False)[:200]}")

    for name in TARGETED_MODULES:
        mod = importlib.import_module(name)
        result = mod.compute(track)   # default target = most-dwelled / every AOI
        result.save(track.sortie, track.seat)
        summary["biomarkers"][name] = result.metrics
        print(f"  [{track.seat}] {name}: {json.dumps(result.metrics, ensure_ascii=False)[:200]}")

    return summary


def run_all(video: str, hz: float = 5.0, aoi_k: int = 6) -> dict:
    print(f"[1/2] extracting gaze ({hz} Hz) ...")
    tracks = extract_gaze.extract(video, hz=hz)
    if not tracks:
        print("no faces detected -- nothing to analyze")
        return {}

    print("[2/2] biomarkers ...")
    sortie = Path(video).stem
    out = {"sortie": sortie, "seats": {}}
    for track in tracks:
        track.save(GAZE_DIR / sortie / f"{track.seat}.json")   # persist raw track too
        out["seats"][track.seat] = run_one_track(track, aoi_k=aoi_k)

    summary_path = GAZE_DIR / sortie / "_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {summary_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--aoi-k", type=int, default=6)
    args = ap.parse_args()
    run_all(args.video, hz=args.hz, aoi_k=args.aoi_k)


if __name__ == "__main__":
    main()
