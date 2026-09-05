"""One-time, per-sortie-per-seat AOI calibration: attach a real name
("engine gauges", "windscreen", ...) to the unsupervised gaze clusters
every other module in this package works with (see schema.py's docstring
for why this can't be automatic -- the camera watches the crew, not what
the crew sees).

Two steps:
  1. --dump-frames VIDEO : for every AOI cluster, saves one representative
     video frame (the moment gaze was closest to that cluster's centroid)
     to exports/gaze/<sortie>/<seat>/aoi_frames/aoi_<n>.png -- look at
     where the pilot's eyes/head are pointed in each to figure out what
     they were probably looking at.
  2. --set "0=engine gauges;3=windscreen;5=other crew member" : writes
     exports/gaze/<sortie>/<seat>/aoi_labels.json. Unlisted AOIs keep
     their generic AOI_<n> id. Prefix a label "in:"/"out:" (e.g.
     "in: engine gauges") to also feed time_inside_vs_outside_panel.py.

Every AOI-consuming module re-reads this file each run (base.load_aoi_labels)
-- no re-processing of the gaze track needed after labeling.

    python3 code/gaze/aoi_labels.py exports/gaze/"s1 c2"/left.json --dump-frames "/path/s1 c2.mp4"
    python3 code/gaze/aoi_labels.py exports/gaze/"s1 c2"/left.json --set "0=in: engine gauges;2=out: windscreen"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import GAZE_DIR, assign_aoi  # noqa: E402
from schema import GazeTrack  # noqa: E402


def dump_frames(track: GazeTrack, video: str, centroids: list[list[float]]) -> Path:
    import cv2
    cap = cv2.VideoCapture(video)
    out_dir = GAZE_DIR / track.sortie / track.seat / "aoi_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = np.array([[s.yaw, s.pitch] for s in track.samples])
    for i, c in enumerate(centroids):
        d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
        best = track.samples[int(np.argmin(d))]
        cap.set(cv2.CAP_PROP_POS_MSEC, best.t * 1000)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out_dir / f"aoi_{i}.png"), frame)
    cap.release()
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gaze_json")
    ap.add_argument("--aoi-k", type=int, default=6)
    ap.add_argument("--dump-frames", metavar="VIDEO", default=None)
    ap.add_argument("--set", metavar="0=label;1=label", default=None)
    args = ap.parse_args()

    track = GazeTrack.load(args.gaze_json)
    info = assign_aoi(track, k=args.aoi_k)
    print("AOI clusters (id: centroid yaw,pitch  size):")
    for i, (c, n) in enumerate(zip(info["centroids"], info["sizes"])):
        print(f"  {i}: yaw={c[0]:+.1f} pitch={c[1]:+.1f}  n={n}")

    if args.dump_frames:
        out_dir = dump_frames(track, args.dump_frames, info["centroids"])
        print(f"Wrote representative frames -> {out_dir}")

    if args.set:
        labels = {}
        for pair in args.set.split(";"):
            k, _, v = pair.partition("=")
            labels[k.strip()] = v.strip()
        out = GAZE_DIR / track.sortie / track.seat / "aoi_labels.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
