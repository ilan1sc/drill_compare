"""11.1 Saccade frequency/velocity, where measurement quality supports it.

The "where quality supports it" qualifier matters here specifically: at
5Hz (extract_gaze.py's default --hz) a real saccade (tens of ms) is
usually captured as a single large jump between two samples, so peak
velocity is systematically underestimated and frequency can be
undercounted (two quick saccades within one 0.2s gap collapse into one).
This module reports its own sample-rate ceiling alongside the numbers,
and low_confidence=True whenever sample_hz < 10, rather than presenting
a velocity figure as trustworthy by default.

Metrics, from base.fixations()'s 'saccade' segments:
    count, per-minute rate, and the angular velocity distribution
    (mean/median/p95 deg/s, computed over base.angular_velocity()'s raw
    consecutive-sample speeds that fall inside a saccade segment).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, angular_velocity, fixations, robust_threshold, run_cli  # noqa: E402


def compute(track) -> Result:
    fx = fixations(track)
    saccades = [f for f in fx if f["kind"] == "saccade"]
    minutes = max(track.duration / 60.0, 1e-9)

    vel = angular_velocity(track)
    thresh = robust_threshold(np.array([v[2] for v in vel])) if vel else 0.0
    sacc_speeds = np.array([v[2] for v in vel if v[2] > thresh])

    metrics = {
        "sample_hz": track.sample_hz,
        "low_confidence": track.sample_hz < 10,
        "saccades": len(saccades),
        "saccades_per_min": round(len(saccades) / minutes, 2),
        "velocity_deg_s": {
            "mean": round(float(sacc_speeds.mean()), 1) if len(sacc_speeds) else None,
            "median": round(float(np.median(sacc_speeds)), 1) if len(sacc_speeds) else None,
            "p95": round(float(np.percentile(sacc_speeds, 95)), 1) if len(sacc_speeds) else None,
        },
    }
    # saccade amplitude = angular distance covered by the segment
    amps = [float(np.hypot(saccades[i]["yaw"] - fx[fx.index(saccades[i]) - 1]["yaw"],
                          saccades[i]["pitch"] - fx[fx.index(saccades[i]) - 1]["pitch"]))
           for i in range(len(saccades)) if fx.index(saccades[i]) > 0]
    metrics["mean_amplitude_deg"] = round(float(np.mean(amps)), 2) if amps else None

    return Result("saccade_frequency_velocity", metrics, saccades[:300],
                 limitations="sample-rate-limited; see module docstring, check low_confidence")


if __name__ == "__main__":
    run_cli("saccade_frequency_velocity", compute)
