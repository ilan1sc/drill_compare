"""11.1 Head/eye stability.

For every fixation (base.fixations), measures how much residual jitter
remained in (a) the combined gaze signal and (b) the head-pose-only
signal (head_yaw/head_pitch, before the eye-in-head offset is added) --
extract_gaze.py records both. A fixation held steady mostly by the eyes
(low head jitter, head doing little) vs. one held steady by re-aiming the
whole head reads very differently even at the same combined-gaze
stability, and this is the one module that can tell them apart.

    stability = 1 / (1 + RMS-of-samples-around-the-fixation-mean)

reported separately for combined gaze and head-only, plus their ratio
(close to 1 = stability is coming mostly from the head; lower = the eyes
are doing more of the steadying).

Limitation: at 5Hz, "jitter" here is frame-to-frame residual after I-VT
segmentation, not the fine physiological head/eye tremor a real
head-tracker + eye-tracker rig would resolve.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, fixations, run_cli  # noqa: E402
from schema import GazeTrack  # noqa: E402


def compute(track: GazeTrack) -> Result:
    fx = fixations(track)
    by_t = {round(p.t, 3): p for p in track.samples}

    rows = []
    for f in fx:
        if f["kind"] != "fixation":
            continue
        pts = [by_t[round(t, 3)] for t in np.arange(f["t0"], f["t1"] + 1e-6, 1 / track.sample_hz)
              if round(t, 3) in by_t]
        if len(pts) < 3:
            continue
        gaze_rms = float(np.hypot(np.std([p.yaw for p in pts]), np.std([p.pitch for p in pts])))
        head_rms = float(np.hypot(np.std([p.head_yaw for p in pts]),
                                  np.std([p.head_pitch for p in pts])))
        rows.append({"t0": f["t0"], "t1": f["t1"], "gaze_rms": round(gaze_rms, 3),
                    "head_rms": round(head_rms, 3),
                    "gaze_stability": round(1 / (1 + gaze_rms), 3),
                    "head_stability": round(1 / (1 + head_rms), 3)})

    if rows:
        g = np.array([r["gaze_stability"] for r in rows])
        h = np.array([r["head_stability"] for r in rows])
        metrics = {
            "fixations_analyzed": len(rows),
            "mean_gaze_stability": round(float(g.mean()), 3),
            "mean_head_stability": round(float(h.mean()), 3),
            "head_vs_gaze_ratio": round(float(h.mean() / (g.mean() or 1e-9)), 3),
        }
    else:
        metrics = {"fixations_analyzed": 0}
    return Result("head_eye_stability", metrics, rows[:300],
                 limitations="frame-to-frame residual at 5Hz, not physiological tremor; "
                             "see module docstring")


if __name__ == "__main__":
    run_cli("head_eye_stability", compute)
