"""11.1 Fixation duration and frequency.

Runs base.fixations() (I-VT, this track's own velocity threshold) and
reports duration statistics and rate. A high fixations/min with short
mean duration reads as scattered/hurried scanning; long mean duration
with few fixations/min reads as fixed, sustained attention.

Limitation: I-VT on a 5Hz appearance-based signal, not a calibrated
eye-tracker -- true microsaccades within a fixation are invisible at this
sample rate, so durations run long relative to lab eye-tracking norms;
compare sorties/pilots to each other under this same pipeline, not to
published eye-tracking literature values.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, fixations, run_cli  # noqa: E402


def compute(track) -> Result:
    fx = [f for f in fixations(track) if f["kind"] == "fixation"]
    minutes = max(track.duration / 60.0, 1e-9)
    durs = np.array([f["dur"] for f in fx])

    metrics = {
        "fixations": len(fx),
        "fixations_per_min": round(len(fx) / minutes, 2),
        "mean_duration_s": round(float(durs.mean()), 3) if len(durs) else None,
        "median_duration_s": round(float(np.median(durs)), 3) if len(durs) else None,
        "p90_duration_s": round(float(np.percentile(durs, 90)), 3) if len(durs) else None,
        "total_fixation_time_s": round(float(durs.sum()), 1) if len(durs) else 0.0,
        "fixation_time_share": round(float(durs.sum()) / track.duration, 3) if len(durs) else 0.0,
    }
    return Result("fixation_duration_frequency", metrics, fx[:300],
                 limitations="I-VT over a 5Hz proxy signal; see module docstring")


if __name__ == "__main__":
    run_cli("fixation_duration_frequency", compute)
