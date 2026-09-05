"""11.1 Scan path and scan-pattern regularity.

Regularity is read off the AOI transition sequence (base.fixations +
base.assign_aoi), not raw gaze coordinates:
  - transition_entropy_norm : Shannon entropy of the AOI-to-AOI edge
    distribution, normalized to 0..1 by log(unique edges). Low = a small
    repertoire of transitions used over and over (a disciplined,
    repeatable scan pattern e.g. instrument -> outside -> instrument);
    high = transitions are scattered across many different edges (an
    unstructured or exploratory scan).
  - edge_repeat_rate : fraction of consecutive transitions that repeat
    the SAME edge back-to-back (A->B, A->B, ...) -- literal habitual
    back-and-forth scanning.
  - mean_path_amplitude_deg : mean angular distance per transition (how
    far the eyes/head travel per hop -- a tight scan visits AOIs close
    to each other, a wide one sweeps the whole field of view).

Limitation: "regular" here is a statistical property (repeatable,
low-entropy), not a judgment of whether the pattern was the RIGHT one for
the maneuver being flown -- pair with pfd_compare_csv.py's drills to see
which maneuver each regularity score belongs to.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, fixations, run_cli  # noqa: E402


def compute(track) -> Result:
    fx = [f for f in fixations(track) if f["kind"] == "fixation" and f["aoi"] is not None]
    seq = [(f["aoi"], f["yaw"], f["pitch"]) for f in fx]
    edges = [(seq[i][0], seq[i + 1][0]) for i in range(len(seq) - 1) if seq[i][0] != seq[i + 1][0]]

    if edges:
        from collections import Counter
        counts = np.array(list(Counter(edges).values()), dtype=float)
        p = counts / counts.sum()
        entropy = float(-(p * np.log(p)).sum())
        norm_entropy = entropy / np.log(len(counts)) if len(counts) > 1 else 0.0
        repeats = sum(1 for i in range(1, len(edges)) if edges[i] == edges[i - 1])
        repeat_rate = repeats / max(len(edges) - 1, 1)
    else:
        norm_entropy = repeat_rate = None

    amps = [float(np.hypot(seq[i + 1][1] - seq[i][1], seq[i + 1][2] - seq[i][2]))
           for i in range(len(seq) - 1)]

    metrics = {
        "fixations": len(fx), "transitions": len(edges),
        "transition_entropy_norm": round(norm_entropy, 3) if norm_entropy is not None else None,
        "edge_repeat_rate": round(repeat_rate, 3) if repeat_rate is not None else None,
        "mean_path_amplitude_deg": round(float(np.mean(amps)), 2) if amps else None,
    }
    return Result("scan_path_regularity", metrics, [],
                 limitations="statistical repeatability, not maneuver-appropriateness; "
                             "see module docstring")


if __name__ == "__main__":
    run_cli("scan_path_regularity", compute, needs_aoi=True)
