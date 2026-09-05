"""10. Overshoot / undershoot cycles -- hunting around the target state
before it settles.

For every move-segment -> hold-segment pair (base.activity_segments), the
hold's own mean is taken as the settled/target value (no external target
is logged). Overshoot cycles = how many times the resulting state crosses
that settled value, alternating sign, between the start of the move and
the point it finally stays inside the hold -- 0 crossings = the state
arrived and stayed, 1 = it passed the target once and came back (a single
overshoot), 2+ = it hunted back and forth. Overshoot magnitude = the
largest excursion past the settled value, as a fraction of the total
commanded change (state at move-start to settled value).

Limitation: "settled value" is the hold's own mean, so a hold that itself
drifts gives a noisy target; short holds (<1.5s, base's min_len_s) are
excluded from being counted as settle points for this reason.
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

        cycles_list, mags = [], []
        for k in range(1, len(segs)):
            kind, i, j = segs[k]
            pkind, pi, pj = segs[k - 1]
            if kind != "hold" or pkind != "move":
                continue
            settle = float(np.mean(state[i:j]))
            window = state[pi:i] - settle
            if len(window) < 3:
                continue
            signs = np.sign(window)
            signs = signs[signs != 0]
            crossings = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
            commanded = abs(state[pi] - settle) or 1e-6
            overshoot_mag = float(np.max(np.abs(window)) / commanded)
            cycles_list.append(crossings)
            mags.append(overshoot_mag)
            if crossings and len(evidence) < 200:
                evidence.append({"channel": name, "t_settle": round(flight["t"][i], 1),
                                 "crossings": crossings, "overshoot_ratio": round(overshoot_mag, 2)})

        c, m = np.array(cycles_list), np.array(mags)
        per_channel[name] = {
            "transitions": len(cycles_list),
            "mean_crossings": round(float(c.mean()), 2) if len(c) else None,
            "clean_arrival_rate": round(float(np.mean(c == 0)), 3) if len(c) else None,
            "mean_overshoot_ratio": round(float(m.mean()), 2) if len(m) else None,
        }

    metrics = {"per_channel": per_channel}
    return Result("controls_overshoot", metrics, evidence,
                 limitations="settled value = the hold's own mean, no external target; "
                             "see module docstring")


if __name__ == "__main__":
    run_cli("controls_overshoot", compute)
