"""10. Stabilization time -- time from the first corrective action until
the aircraft returns to, and stays in, an acceptable stable region.

For every move-segment -> hold-segment pair (base.activity_segments), the
stable band is +/- 2 standard deviations of the state DURING the
following hold (i.e. however tightly the cadet actually held it,
scenario-relative rather than a fixed tolerance -- see controls_precision.py
for the raw hold tightness itself). stabilization_time is measured from
the control's first movement to the LAST moment the state was still
outside that band -- i.e. the point after which it never leaves the band
again before/through the hold.

Limitation: uses the following hold's own spread to define "acceptable",
so a sloppy hold produces a wide band and an easy (fast) stabilization
number -- read together with controls_precision.py, not alone.
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
        state = flight[spec["state"]]
        thresh = robust_threshold(rate)
        segs = activity_segments(rate, flight["t"], thresh, min_len_s=1.0)

        times = []
        for k in range(1, len(segs)):
            kind, i, j = segs[k]
            pkind, pi, pj = segs[k - 1]
            if kind != "hold" or pkind != "move":
                continue
            hold_state = state[i:j]
            settle = float(np.mean(hold_state))
            band = max(2.0 * float(np.std(hold_state)), 1e-6)

            window = state[pi:j]
            outside = np.where(np.abs(window - settle) > band)[0]
            stab_idx = pi + (outside[-1] + 1 if len(outside) else 0)
            stab_idx = min(stab_idx, j - 1)
            t_stab = flight["t"][stab_idx] - flight["t"][pi]
            times.append(t_stab)
            if len(evidence) < 200:
                evidence.append({"channel": name, "t_move_start": round(flight["t"][pi], 1),
                                 "stabilization_time_s": round(float(t_stab), 2),
                                 "band": round(band, 3)})

        arr = np.array(times)
        per_channel[name] = {
            "transitions": len(arr),
            "mean_stabilization_s": round(float(arr.mean()), 2) if len(arr) else None,
            "median_stabilization_s": round(float(np.median(arr)), 2) if len(arr) else None,
            "p90_stabilization_s": round(float(np.percentile(arr, 90)), 2) if len(arr) else None,
        }

    metrics = {"per_channel": per_channel}
    return Result("controls_stabilization_time", metrics, evidence,
                 limitations="stable band = the following hold's own spread; see module docstring")


if __name__ == "__main__":
    run_cli("controls_stabilization_time", compute)
