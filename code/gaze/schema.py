"""Shared data shape for the video/gaze pipeline (PRD 11.1).

IMPORTANT scope note, read before trusting any AOI label downstream:
the face camera (`s1 cX.mp4`, "COPILOT CAM COCKPIT") is mounted in the
cabin looking AT the crew -- it is not a POV/eye-tracker-glasses feed
looking at what the pilot sees. So this pipeline can recover WHERE a
pilot is looking in angular terms (gaze yaw/pitch relative to their own
head) and cluster those directions into areas of interest, but it cannot
know which physical instrument a given AOI cluster corresponds to
without a one-time human calibration (see aoi.py's `--label` mechanism).
Every AOI-based module says this again locally; it is the single biggest
caveat on this whole engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GazeSample:
    t: float
    seat: str            # "left" / "right" -- which crew position, by face x-position
    yaw: float            # combined head+eye gaze, degrees, camera-relative (+ = subject's right)
    pitch: float          # degrees (+ = up)
    head_yaw: float
    head_pitch: float
    conf: float           # 0..1, detection quality proxy
    aoi: int | None = None            # filled in by aoi.py
    aoi_label: str | None = None      # filled in once a human calibrates (aoi.py --label)


@dataclass
class GazeTrack:
    sortie: str
    source_video: str
    seat: str              # this track is single-seat: "left" or "right"
    sample_hz: float
    duration: float
    samples: list[GazeSample] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: str | Path) -> "GazeTrack":
        d = json.loads(Path(path).read_text())
        samples = [GazeSample(**s) for s in d["samples"]]
        return GazeTrack(sortie=d["sortie"], source_video=d["source_video"], seat=d["seat"],
                         sample_hz=d["sample_hz"], duration=d["duration"], samples=samples)
