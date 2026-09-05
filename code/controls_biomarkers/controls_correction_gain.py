"""10. Correction gain -- proportionality between deviation magnitude and
corrective control input.

Treats the cadet as a proportional controller during hold segments
(base.activity_segments): deviation(t) = state(t) - hold_mean is the
error from what they're trying to hold, and control_rate(t + lag) is
their correction. For each channel, searches a small reaction-lag window
(0..1.5s, same lag-search idiom as pfd_compare_csv3.py/csv4.py) for the
lag that best correlates deviation with the following control_rate, then
fits control_rate ~ gain * deviation + b by least squares at that lag.

    gain  : gain of the fitted line -- how much the cadet nudges the
            control per unit of deviation (bigger |gain| = more reactive
            per unit error)
    r     : how proportional/consistent that response is (near 0 = the
            corrective input isn't really tracking the deviation size)
    lag_s : the reaction lag that fit best

Limitation: only measured during holds (small deviations, low control
activity); large active corrections during a maneuver are a different
regime and are NOT included here -- see controls_stabilization_time.py
and controls_aggressiveness.py for those.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import CHANNELS, Result, activity_segments, robust_threshold, run_cli  # noqa: E402


def _fit(dev, rate):
    if len(dev) < 20 or np.std(dev) == 0:
        return None
    gain, b = np.polyfit(dev, rate, 1)
    pred = gain * dev + b
    r = np.corrcoef(pred, rate)[0, 1] if np.std(pred) > 0 else 0.0
    return float(gain), float(b), float(r)


def compute(flight) -> Result:
    per_channel = {}
    for name, spec in CHANNELS.items():
        rate = flight[f"{name}_rate"]
        state = flight[spec["state"]]
        thresh = robust_threshold(rate)
        segs = activity_segments(rate, flight["t"], thresh, min_len_s=1.5)
        holds = [(i, j) for kind, i, j in segs if kind == "hold"]

        devs, rates_by_lag = [], {}
        lags = np.arange(0, 1.6, flight["dt"] * 3)
        best = None
        for lag_s in lags:
            lag = int(round(lag_s / flight["dt"]))
            dev_all, rate_all = [], []
            for i, j in holds:
                if j - lag <= i:
                    continue
                seg_state = state[i:j]
                dev = seg_state - seg_state.mean()
                dev_all.append(dev[: len(dev) - lag] if lag else dev)
                rate_all.append(rate[i + lag:j] if lag else rate[i:j])
            if not dev_all:
                continue
            dev_all = np.concatenate(dev_all)
            rate_all = np.concatenate(rate_all)
            fit = _fit(dev_all, rate_all)
            if fit and (best is None or abs(fit[2]) > abs(best[3])):
                best = (lag_s, *fit)

        if best:
            lag_s, gain, b, r = best
            per_channel[name] = {"gain": round(gain, 4), "r": round(r, 3),
                                 "lag_s": round(float(lag_s), 2), "holds_used": len(holds)}
        else:
            per_channel[name] = {"gain": None, "r": None, "lag_s": None, "holds_used": len(holds)}

    metrics = {"per_channel": per_channel}
    return Result("controls_correction_gain", metrics, [],
                 limitations="proportional-response fit over hold segments only; "
                             "see module docstring")


if __name__ == "__main__":
    run_cli("controls_correction_gain", compute)
