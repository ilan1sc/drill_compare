"""10. Control smoothness -- rate, jerk, oscillation and abruptness of
control inputs.

Per channel (collective / lateral cyclic / longitudinal cyclic):
  - rate       : |d(control)/dt|, RMS -- how fast the control moves overall
  - jerk       : |d2(control)/dt2|, RMS -- how abruptly the RATE itself changes;
                the classic smoothness signal (a jerky hand produces spiky jerk,
                a smooth hand produces a gently-varying rate)
  - oscillation: reversals in control direction per minute (zero-crossings
                of the rate signal) -- a twitchy/hunting input reverses
                far more often than a deliberate one
  - abruptness : ratio of the 95th-percentile input rate to the median
                input rate -- how much bigger the "big" movements are
                than typical ones

Limitation: pure signal statistics on the raw control channel; doesn't
know which movements were maneuver-appropriate (a steep turn entry
*should* have a bigger rate spike than cruise) -- pair with
pfd_sync/pfd_compare_csv.py's drills to view this per maneuver type.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import CHANNELS, Result, run_cli  # noqa: E402


def compute(flight) -> Result:
    per_channel = {}
    for name, spec in CHANNELS.items():
        rate = flight[f"{name}_rate"]
        jerk = np.gradient(rate, flight["dt"])
        crossings = int(np.sum(np.diff(np.sign(rate)) != 0))
        minutes = (flight["t"][-1] - flight["t"][0]) / 60.0
        p95 = float(np.percentile(np.abs(rate), 95))
        # a channel held nearly dead-still (median rate ~0, e.g. collective in
        # cruise) makes p95/median blow up; floor the denominator at 1% of p95
        # itself so the ratio stays a bounded "how much bigger are the big
        # moves than typical" read rather than a division-by-near-zero spike
        med_rate = max(float(np.median(np.abs(rate))), p95 * 0.01, 1e-9)

        per_channel[name] = {
            "rate_rms": round(float(np.sqrt(np.mean(rate ** 2))), 4),
            "jerk_rms": round(float(np.sqrt(np.mean(jerk ** 2))), 4),
            "oscillations_per_min": round(crossings / minutes, 2),
            "abruptness_ratio": round(p95 / med_rate, 2),
            "unit": spec["ctrl_unit"] + "/s",
        }

    metrics = {"per_channel": per_channel}
    return Result("controls_smoothness", metrics, [],
                 limitations="raw signal statistics, maneuver-agnostic; see module docstring")


if __name__ == "__main__":
    run_cli("controls_smoothness", compute)
