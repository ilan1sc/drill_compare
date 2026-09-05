"""11.1 Time inside vs. outside the instrument panel.

No calibration = no known panel boundary, so this uses a cockpit-geometry
heuristic by default: the panel sits below eye line, so pitch below
--pitch-split (default: this track's own median pitch, i.e. "the bottom
half of where this pilot actually looks" -- scenario-relative again) is
treated as "inside" (panel/console) and the rest as "outside" (windscreen
/ scanning outside, or looking at the other crew member).

If code/gaze/aoi_labels.py has been run and an AOI's label starts with
"in:" or "out:" (e.g. "in: engine gauges", "out: windscreen"), those
labeled AOIs override the pitch heuristic for their samples -- the more
AOIs get calibrated this way, the less this module depends on the
heuristic at all (reported as calibrated_share).

Limitation: the default split is a geometric guess, not a measured panel
edge -- treat the *trend* across a sortie or between sorties as more
meaningful than the absolute inside/outside split on any one of them.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, load_aoi_labels, run_cli  # noqa: E402


def compute(track, pitch_split: float | None = None) -> Result:
    s = track.samples
    if not s:
        return Result("time_inside_vs_outside_panel", {}, [], "no samples")

    labels = load_aoi_labels(track.sortie, track.seat) or {}
    pitches = np.array([p.pitch for p in s])
    split = pitch_split if pitch_split is not None else float(np.median(pitches))

    inside = calibrated = 0
    for p in s:
        lab = labels.get(str(p.aoi)) if p.aoi is not None else None
        if lab and (lab.startswith("in:") or lab.startswith("out:")):
            calibrated += 1
            inside += lab.startswith("in:")
        else:
            inside += p.pitch < split

    n = len(s)
    metrics = {
        "pitch_split_deg": round(split, 2),
        "calibrated_share": round(calibrated / n, 3),
        "inside_panel_share": round(inside / n, 3),
        "outside_panel_share": round(1 - inside / n, 3),
    }
    return Result("time_inside_vs_outside_panel", metrics, [],
                 limitations="pitch-split heuristic unless AOIs are calibrated in:/out:; "
                             "see module docstring")


if __name__ == "__main__":
    run_cli("time_inside_vs_outside_panel", compute, needs_aoi=True)
