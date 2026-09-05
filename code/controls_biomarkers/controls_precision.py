"""10. Control precision -- accuracy of control application relative to
the required maneuver.

No target/ATP values are logged, so precision is measured against the
cadet's OWN intent: whenever a control is held steady (its rate below
this sortie's own noise floor, base.activity_segments) the cadet is
implicitly asking the aircraft to hold whatever state that produced.
Precision = how tightly the resulting state (VS / bank / pitch) actually
stayed put during that hold -- its standard deviation. Lower = more
precise. Reported per channel, in the state's own units, plus a
normalized 0..1 score (1 = dead steady) for cross-channel comparison.

Limitation: a hold with the aircraft was drifting for an EXTERNAL reason
(turbulence, a request from the instructor) reads as imprecise even
though the cadet's control input was steady -- this measures resulting
state stability, not control-input accuracy in isolation.
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
        segs = activity_segments(rate, flight["t"], thresh, min_len_s=1.5)
        holds = [(i, j) for kind, i, j in segs if kind == "hold"]

        state = flight[spec["state"]]
        stds = []
        for i, j in holds:
            s = state[i:j]
            std = float(np.std(s))
            stds.append(std)
            if len(evidence) < 200:
                evidence.append({"channel": name, "t0": round(flight["t"][i], 1),
                                 "t1": round(flight["t"][j - 1], 1),
                                 "hold_std": round(std, 3)})
        if stds:
            arr = np.array(stds)
            # normalize: 1.0 at the sortie's best (10th pct) hold, decaying with worse spread
            scale = max(np.percentile(arr, 10), 1e-6)
            score = float(np.mean(1.0 / (1.0 + arr / scale)))
        else:
            arr, score = np.array([]), None

        per_channel[name] = {
            "holds": len(holds),
            "mean_hold_std": round(float(arr.mean()), 3) if len(arr) else None,
            "median_hold_std": round(float(np.median(arr)), 3) if len(arr) else None,
            "unit": spec["state_unit"],
            "precision_score_0_1": round(score, 3) if score is not None else None,
        }

    metrics = {"per_channel": per_channel}
    return Result("controls_precision", metrics, evidence,
                 limitations="holds vs. own control-rate baseline; see module docstring")


if __name__ == "__main__":
    run_cli("controls_precision", compute)
