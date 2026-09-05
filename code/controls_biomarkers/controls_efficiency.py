"""10. Control efficiency / minimum movement -- how much control travel
was spent to get from one held position to the next.

For every "move" segment between two holds (base.activity_segments):
    path_length = integral of |control_rate| dt   (total distance the
                  control actually travelled, back-and-forth included)
    net_move    = |control(end) - control(start)| (straight-line distance)
    efficiency  = net_move / path_length           (1.0 = moved straight
                  to the new position with no backtracking or hunting;
                  lower = wasted motion)

Limitation: a deliberately curved/staged input (e.g. easing into a turn)
is not "inefficient" in the way a beginner's overcorrection is, but both
score the same here -- read alongside controls_correction_count.py and
controls_overshoot.py before calling a low score a problem.
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

        effs = []
        for i, j in moves:
            path = float(np.sum(np.abs(rate[i:j])) * flight["dt"])
            net = float(abs(ctrl[j - 1] - ctrl[i]))
            if path > 1e-6:
                eff = min(net / path, 1.0)
                effs.append(eff)
                if len(evidence) < 200:
                    evidence.append({"channel": name, "t0": round(flight["t"][i], 1),
                                     "t1": round(flight["t"][j - 1], 1),
                                     "net_move": round(net, 3), "path_length": round(path, 3),
                                     "efficiency": round(eff, 3)})

        arr = np.array(effs)
        per_channel[name] = {
            "moves": len(moves),
            "mean_efficiency": round(float(arr.mean()), 3) if len(arr) else None,
            "median_efficiency": round(float(np.median(arr)), 3) if len(arr) else None,
        }

    metrics = {"per_channel": per_channel}
    return Result("controls_efficiency", metrics, evidence,
                 limitations="path-length vs. net-displacement ratio; see module docstring")


if __name__ == "__main__":
    run_cli("controls_efficiency", compute)
