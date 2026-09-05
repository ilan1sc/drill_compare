"""10. Correction count -- number of sub-corrections needed before
stabilization.

Within each "move" segment (base.activity_segments), counts reversals in
control direction (sign changes of control_rate that exceed the sortie's
own noise floor) AFTER the first movement -- i.e. how many times the
cadet changed their mind / re-adjusted before settling, rather than
producing one clean input. 0 extra reversals = a single clean correction.

Limitation: a maneuver that legitimately needs multiple deliberate control
changes (e.g. easing in, holding, easing out of a turn) will also show
reversals here -- this counts direction changes, it doesn't judge whether
they were necessary.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import CHANNELS, Result, activity_segments, robust_threshold, run_cli  # noqa: E402


def compute(flight) -> Result:
    per_channel, evidence = {}, []
    for name, spec in CHANNELS.items():
        rate = flight[f"{name}_rate"]
        thresh = robust_threshold(rate)
        segs = activity_segments(rate, flight["t"], thresh, min_len_s=0.3)
        moves = [(i, j) for kind, i, j in segs if kind == "move"]

        counts = []
        for i, j in moves:
            seg_rate = rate[i:j]
            sign = np.sign(np.where(np.abs(seg_rate) > thresh, seg_rate, 0))
            sign = sign[sign != 0]
            reversals = int(np.sum(np.diff(sign) != 0)) if len(sign) > 1 else 0
            counts.append(reversals)
            if reversals and len(evidence) < 200:
                evidence.append({"channel": name, "t0": round(flight["t"][i], 1),
                                 "t1": round(flight["t"][j - 1], 1), "reversals": reversals})

        arr = np.array(counts)
        per_channel[name] = {
            "maneuvers": len(moves),
            "mean_corrections": round(float(arr.mean()), 2) if len(arr) else None,
            "clean_single_input_rate": round(float(np.mean(arr == 0)), 3) if len(arr) else None,
            "max_corrections": int(arr.max()) if len(arr) else None,
        }

    metrics = {"per_channel": per_channel}
    return Result("controls_correction_count", metrics, evidence,
                 limitations="direction reversals above the sortie's own noise floor; "
                             "see module docstring")


if __name__ == "__main__":
    run_cli("controls_correction_count", compute)
