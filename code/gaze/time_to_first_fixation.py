"""11.1 Time to first fixation on a relevant instrument/AOI.

Without --events-csv, "relevant" defaults to every AOI and the reference
point is the start of the recording -- i.e. this reports how long it took
the pilot to first look at each AOI cluster at all after the video began
(mostly useful as a sanity check / for the busiest AOIs).

The intended real use is with --events-csv (columns start,end,kind --
e.g. the drills table from pfd_compare_csv.py, re-timed to video time via
pfd_compare_csv3.py/csv4.py): for --target-aoi and every event, time to
first fixation on the target AFTER that event's start -- e.g. "how long
after entering a steep turn before the cadet's gaze reached the
attitude/bank AOI".

Limitation: needs the drills CSV in this SAME video's time base to be
useful for anything but "since the recording started" -- see the two
alignment tools above for getting a data-time drills table onto video time.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, assign_aoi, fixations  # noqa: E402
from schema import GazeTrack  # noqa: E402


def compute(track, target_aoi=None, events_csv: str | None = None) -> Result:
    fx = [f for f in fixations(track) if f["kind"] == "fixation" and f["aoi"] is not None]
    if not fx:
        return Result("time_to_first_fixation", {}, [], "no fixations")

    targets = [target_aoi] if target_aoi is not None else sorted({f["aoi"] for f in fx})
    events = [(0.0, "sortie_start")]
    if events_csv:
        with open(events_csv) as fh:
            for row in csv.DictReader(fh):
                events.append((float(row["start"]), row.get("kind") or row.get("label") or "event"))

    rows = []
    for t_aoi in targets:
        times = [f["t0"] for f in fx if f["aoi"] == t_aoi]
        label = next((f["aoi_label"] for f in fx if f["aoi"] == t_aoi), None)
        for ev_t, ev_kind in events:
            after = [t for t in times if t >= ev_t]
            latency = round(after[0] - ev_t, 2) if after else None
            rows.append({"aoi": t_aoi, "aoi_label": label, "event": ev_kind,
                        "event_t": round(ev_t, 1), "latency_s": latency})

    latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    metrics = {
        "events": len(events), "targets": len(targets),
        "mean_latency_s": round(float(np.mean(latencies)), 2) if latencies else None,
        "median_latency_s": round(float(np.median(latencies)), 2) if latencies else None,
        "never_reached_rate": round(1 - len(latencies) / max(len(rows), 1), 3),
    }
    return Result("time_to_first_fixation", metrics, rows[:300],
                 limitations="uses sortie start unless --events-csv given; see module docstring")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gaze_json")
    ap.add_argument("--target-aoi", default=None)
    ap.add_argument("--events-csv", default=None)
    ap.add_argument("--aoi-k", type=int, default=6)
    args = ap.parse_args()

    track = GazeTrack.load(args.gaze_json)
    if not track.samples or track.samples[0].aoi is None:
        assign_aoi(track, k=args.aoi_k)
    target = int(args.target_aoi) if args.target_aoi and args.target_aoi.lstrip("-").isdigit() \
        else args.target_aoi
    result = compute(track, target_aoi=target, events_csv=args.events_csv)
    out = result.save(track.sortie, track.seat)
    print(f"time_to_first_fixation: {result.metrics}")
    print(f"Wrote {out}")
