"""10. Anticipation -- control initiated in advance of a predictable
requirement rather than only after an error already appears.

For every move segment (base.activity_segments; onset = flight["t"][i],
the moment the control starts moving) the resulting state is scanned in
a +/-3s window around that onset for the moment it first visibly departs
from its pre-move baseline (exceeds this sortie's own state-noise floor,
robust_threshold on the state's own rate of change).

    anticipation_lag_s = (state deviation onset) - (control onset)

Positive = the control started moving BEFORE the state had visibly begun
to change (anticipatory / preemptive). Negative = the state had already
started drifting before the cadet reacted (purely reactive, error-driven).
Near zero = essentially simultaneous (expected for a tightly-coupled axis
like lateral cyclic -> roll, which responds almost immediately).

Limitation: this can't distinguish "anticipated a predictable requirement"
from "over-controlled a channel with no state change coming" -- a large
positive lag with no state change ever materializing isn't in this count
(no crossing found -> excluded), but a lag driven by noise in a busy
channel will still show up as a small anticipation value.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import CHANNELS, Result, activity_segments, robust_threshold, run_cli  # noqa: E402

WINDOW_S = 3.0


def compute(flight) -> Result:
    per_channel, evidence = {}, []
    for name, spec in CHANNELS.items():
        rate = flight[f"{name}_rate"]
        state = flight[spec["state"]]
        dt = flight["dt"]
        state_rate_thresh = robust_threshold(np.gradient(state, dt))
        thresh = robust_threshold(rate)
        segs = activity_segments(rate, flight["t"], thresh, min_len_s=0.3)
        moves = [(i, j) for kind, i, j in segs if kind == "move"]

        lags = []
        win = int(WINDOW_S / dt)
        for i, j in moves:
            lo, hi = max(0, i - win), min(len(state), i + win)
            baseline = state[max(0, i - int(1.0 / dt)):i]
            if len(baseline) == 0:
                continue
            base_val = float(np.mean(baseline))
            local_rate = np.abs(np.gradient(state[lo:hi], dt))
            crossing = np.where(local_rate > state_rate_thresh)[0]
            if len(crossing) == 0:
                continue
            onset_idx = lo + crossing[0]
            lag = flight["t"][onset_idx] - flight["t"][i]
            lags.append(lag)
            if len(evidence) < 200:
                evidence.append({"channel": name, "t_control_onset": round(flight["t"][i], 1),
                                 "anticipation_lag_s": round(float(lag), 2)})

        arr = np.array(lags)
        per_channel[name] = {
            "moves_matched": len(arr), "moves_total": len(moves),
            "mean_anticipation_s": round(float(arr.mean()), 2) if len(arr) else None,
            "anticipatory_share": round(float(np.mean(arr > 0.15)), 3) if len(arr) else None,
            "reactive_share": round(float(np.mean(arr < -0.15)), 3) if len(arr) else None,
        }

    metrics = {"per_channel": per_channel}
    return Result("controls_anticipation", metrics, evidence,
                 limitations="control-onset vs. state-deviation-onset lag; see module docstring")


if __name__ == "__main__":
    run_cli("controls_anticipation", compute)
