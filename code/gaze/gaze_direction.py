"""11.1 Gaze direction estimation -- the base signal every other module
in this package builds on.

Not a detector by itself (that's extract_gaze.py, mediapipe head-pose +
iris offset); this reports on the resulting yaw/pitch track: coverage
(how much of the sortie has a usable estimate), its angular range and
spread, and how much of the signal is head rotation vs. eye-in-head
movement (yaw/pitch - head_yaw/head_pitch) -- i.e. does this pilot scan
mostly by turning their head or by moving their eyes.

Limitation: appearance-based (iris-offset) gaze, not a calibrated
eye-tracker -- treat yaw/pitch as a consistent WITHIN-sortie relative
signal, not an absolute angle to any real-world point.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, run_cli  # noqa: E402


def compute(track) -> Result:
    s = track.samples
    expected = max(1, int(track.duration * track.sample_hz))
    if not s:
        return Result("gaze_direction", {"coverage": 0.0, "samples": 0}, [],
                     "no samples -- no face detected in this track")

    yaw = np.array([p.yaw for p in s])
    pitch = np.array([p.pitch for p in s])
    hyaw = np.array([p.head_yaw for p in s])
    hpitch = np.array([p.head_pitch for p in s])
    eye_yaw, eye_pitch = yaw - hyaw, pitch - hpitch

    metrics = {
        "samples": len(s), "coverage": round(len(s) / expected, 3),
        "yaw_range_deg": [round(float(yaw.min()), 1), round(float(yaw.max()), 1)],
        "pitch_range_deg": [round(float(pitch.min()), 1), round(float(pitch.max()), 1)],
        "yaw_std_deg": round(float(yaw.std()), 2), "pitch_std_deg": round(float(pitch.std()), 2),
        "head_contribution": round(float(hyaw.std() / (yaw.std() or 1e-9)), 3),
        "eye_contribution": round(float(eye_yaw.std() / (yaw.std() or 1e-9)), 3),
    }
    return Result("gaze_direction", metrics, [],
                 limitations="appearance-based proxy, within-sortie relative signal only; "
                             "see module docstring")


if __name__ == "__main__":
    run_cli("gaze_direction", compute)
