"""10. Control aggressiveness / conservatism -- characterized RELATIVE to
the scenario and the size of correction actually required, not as a
standalone personality judgment (per PRD 10).

For every move segment (base.activity_segments): size = the net control
travel (how big a correction this was) and peak = the fastest input rate
used getting there. Bigger corrections naturally get faster peak rates,
so this fits peak ~ a * size across ALL of this sortie's own moves per
channel (the cadet's own norm) and scores each move as

    aggressiveness_index = actual_peak / predicted_peak_for_that_size

>1 = punchier than their own typical response to a correction that size
(aggressive), <1 = eased into it more than usual (conservative), ~1 =
typical for them. The sortie-level number is the median index -- values
away from 1 mean the fitted "own norm" line doesn't hold consistently,
not that the cadet is uniformly aggressive or conservative.

Limitation: entirely self-referential (no instructor/ATP standard of
"correct" aggressiveness) -- this describes consistency and relative
style, never an absolute judgment of correctness.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import CHANNELS, Result, activity_segments, robust_threshold, run_cli  # noqa: E402


def compute(flight) -> Result:
    per_channel, evidence = {}, []
    for name, spec in CHANNELS.items():
        ctrl, rate = flight[name], flight[f"{name}_rate"]
        thresh = robust_threshold(rate)
        segs = activity_segments(rate, flight["t"], thresh, min_len_s=0.3)
        moves = [(i, j) for kind, i, j in segs if kind == "move"]

        sizes, peaks, times = [], [], []
        for i, j in moves:
            size = float(abs(ctrl[j - 1] - ctrl[i]))
            peak = float(np.max(np.abs(rate[i:j])))
            if size > 1e-4:
                sizes.append(size)
                peaks.append(peak)
                times.append((flight["t"][i], flight["t"][j - 1]))

        if len(sizes) >= 8:
            a, b = np.polyfit(sizes, peaks, 1)
            predicted = np.maximum(a * np.array(sizes) + b, 1e-6)
            idx = np.array(peaks) / predicted
            for k, (t0, t1) in enumerate(times):
                if len(evidence) < 200:
                    evidence.append({"channel": name, "t0": round(t0, 1), "t1": round(t1, 1),
                                     "size": round(sizes[k], 3), "peak_rate": round(peaks[k], 3),
                                     "aggressiveness_index": round(float(idx[k]), 2)})
            per_channel[name] = {
                "moves": len(sizes), "fit_slope": round(float(a), 4),
                "median_aggressiveness_index": round(float(np.median(idx)), 2),
                "aggressive_share": round(float(np.mean(idx > 1.3)), 3),
                "conservative_share": round(float(np.mean(idx < 0.7)), 3),
            }
        else:
            per_channel[name] = {"moves": len(sizes), "fit_slope": None,
                                 "median_aggressiveness_index": None,
                                 "aggressive_share": None, "conservative_share": None}

    metrics = {"per_channel": per_channel}
    return Result("controls_aggressiveness", metrics, evidence,
                 limitations="self-referential to this sortie's own size-vs-speed norm; "
                             "see module docstring")


if __name__ == "__main__":
    run_cli("controls_aggressiveness", compute)
