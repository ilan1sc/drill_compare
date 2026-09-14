"""Gaze-accuracy lab: run several variants of the face-landmark gaze
extractor over the SAME clip and compare them, to find what actually
moves accuracy for this footage (dim, green-cast cabin IR video).

Baseline problem (measured on exports/gaze/s1 c6/left.json): only ~9%
of sampled frames produce a detection at all. A "gaze estimate" that's
usually just a hole isn't accurate, it's absent -- so the primary metric
here is DETECTION COVERAGE, not just jitter once detected.

Metrics per config, restricted to the LEFT seat (cadet's seat -- the
only one with a usable face view in this camera):
  - coverage   : detections / frames sampled
  - avg_conf   : mean confidence of kept detections (currently constant
                 1.0 upstream -- kept for when that's made real)
  - jitter_deg : median |delta yaw| + |delta pitch| between consecutive
                 KEPT samples -- a precision proxy (lower = steadier),
                 not by itself proof of correctness
  - dropped_implausible : frames where a face was found but yaw/pitch
                 exceeded MAX_PLAUSIBLE_DEG (bad pose decomposition,
                 not real head motion) and were discarded

Usage:
    python3 gazelab2/lab.py --video "/path/s1 c6.mp4" --start 0 --end 300
"""

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR.parent / "code" / "gaze"))
from schema import GazeSample, GazeTrack  # noqa: E402

R_EYE = {"outer": 33, "inner": 133, "top": 159, "bot": 145, "iris": 468}
L_EYE = {"outer": 263, "inner": 362, "top": 386, "bot": 374, "iris": 473}
MAX_PLAUSIBLE_DEG = 85.0
MODEL_PATH = LAB_DIR.parent / "code" / "gaze" / "face_landmarker.task"


@dataclass
class Config:
    name: str
    crop: tuple | None = None          # x0,y0,x1,y1 px, applied before everything else
    upscale: float = 1.0
    enhance: str = "graygrad_clahe"    # "none" | "clahe_only" | "graygrad_clahe"
    clahe_clip: float = 3.0
    min_det_conf: float = 0.3
    min_pres_conf: float = 0.3
    eye_range_deg: float = 20.0
    hz: float = 5.0
    note: str = ""


def _enhance(frame: np.ndarray, mode: str, clahe_clip: float) -> np.ndarray:
    if mode == "none":
        return frame
    f = frame
    if mode == "graygrad_clahe":
        ff = f.astype(np.float32)
        mean = ff.reshape(-1, 3).mean(0)
        ff = ff * (mean.mean() / (mean + 1e-6))
        f = np.clip(ff, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _head_pose(matrix) -> tuple[float, float, float]:
    R = np.array(matrix).reshape(4, 4)[:3, :3]
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(yaw), float(pitch), float(roll)


def _eye_offset(lm, w, h, eye):
    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])
    outer, inner, top, bot, iris = (pt(eye[k]) for k in ("outer", "inner", "top", "bot", "iris"))
    ew, eh = np.linalg.norm(outer - inner), np.linalg.norm(top - bot)
    if ew < 1 or eh < 1:
        return None
    center = (outer + inner) / 2
    dx = (iris[0] - center[0]) / (ew / 2)
    dy = (iris[1] - (top[1] + bot[1]) / 2) / (eh / 2)
    return float(np.clip(dx, -1.5, 1.5)), float(np.clip(dy, -1.5, 1.5))


def run_config(video: Path, cfg: Config, start: float, end: float) -> dict:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = vision.FaceLandmarkerOptions(
        base_options=base, num_faces=1, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=cfg.min_det_conf,
        min_face_presence_confidence=cfg.min_pres_conf,
        running_mode=vision.RunningMode.IMAGE)
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / cfg.hz)))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    samples: list[GazeSample] = []
    n_sampled = 0
    n_detected = 0
    n_dropped_implausible = 0
    t0 = time.time()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = start + idx / fps
        if t > end:
            break
        if idx % step == 0:
            n_sampled += 1
            f = frame
            if cfg.crop is not None:
                x0, y0, x1, y1 = cfg.crop
                f = f[max(0, y0):y1, max(0, x0):x1]
            if cfg.upscale != 1.0:
                f = cv2.resize(f, None, fx=cfg.upscale, fy=cfg.upscale, interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(_enhance(f, cfg.enhance, cfg.clahe_clip), cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.face_landmarks:
                n_detected += 1
                lm = res.face_landmarks[0]
                matrix = res.facial_transformation_matrixes[0]
                hyaw, hpitch, _ = _head_pose(matrix)
                offs = [o for o in (_eye_offset(lm, f.shape[1], f.shape[0], R_EYE),
                                    _eye_offset(lm, f.shape[1], f.shape[0], L_EYE)) if o]
                if offs:
                    dx = float(np.mean([o[0] for o in offs])) * cfg.eye_range_deg
                    dy = float(np.mean([o[1] for o in offs])) * cfg.eye_range_deg
                else:
                    dx = dy = 0.0
                yaw, pitch = hyaw + dx, hpitch + dy
                if abs(yaw) > MAX_PLAUSIBLE_DEG or abs(pitch) > MAX_PLAUSIBLE_DEG:
                    n_dropped_implausible += 1
                else:
                    samples.append(GazeSample(t=round(t, 3), seat="left", yaw=round(yaw, 2),
                                              pitch=round(pitch, 2), head_yaw=round(hyaw, 2),
                                              head_pitch=round(hpitch, 2), conf=1.0))
        idx += 1
    cap.release()
    wall = time.time() - t0

    jitters = []
    for a, b in zip(samples, samples[1:]):
        if b.t - a.t <= 1.0 / cfg.hz * 2.5:
            jitters.append(abs(b.yaw - a.yaw) + abs(b.pitch - a.pitch))
    jitter_deg = statistics.median(jitters) if jitters else float("nan")

    return {
        "config": cfg.name,
        "note": cfg.note,
        "n_sampled": n_sampled,
        "n_detected": n_detected,
        "coverage": n_detected / n_sampled if n_sampled else 0.0,
        "n_dropped_implausible": n_dropped_implausible,
        "jitter_deg_median": jitter_deg,
        "wall_s": round(wall, 1),
        "samples": [s for s in samples],
    }


CONFIGS = [
    Config("A_baseline", crop=None, upscale=1.0, enhance="graygrad_clahe",
           note="current extract_gaze.py defaults, full frame, num_faces would be 2 in prod"),
    Config("B_left_crop", crop=(0, 0, 960, 1080), upscale=1.0, enhance="graygrad_clahe",
           note="isolate left seat half -- drops right-seat face + far-frame clutter"),
    Config("C_crop_upscale2x", crop=(0, 0, 960, 1080), upscale=2.0, enhance="graygrad_clahe",
           note="+ 2x upscale before detection -- small/soft faces detect better enlarged"),
    Config("D_crop_upscale2x_noenhance", crop=(0, 0, 960, 1080), upscale=2.0, enhance="none",
           note="does the gray-world+CLAHE enhancement actually help, or hurt?"),
    Config("E_crop_upscale2x_clahe_only", crop=(0, 0, 960, 1080), upscale=2.0, enhance="clahe_only",
           note="CLAHE contrast without the gray-world white-balance step"),
    Config("F_crop_upscale2x_lowdet", crop=(0, 0, 960, 1080), upscale=2.0, enhance="graygrad_clahe",
           min_det_conf=0.1, min_pres_conf=0.1,
           note="loosen mediapipe's own detection/presence thresholds"),
    Config("G_crop_upscale3x", crop=(0, 0, 960, 1080), upscale=3.0, enhance="graygrad_clahe",
           note="push upscale further -- diminishing returns or does it keep helping?"),
    Config("H_crop_upscale2x_strongclahe", crop=(0, 0, 960, 1080), upscale=2.0,
           enhance="graygrad_clahe", clahe_clip=6.0,
           note="stronger local contrast boost for very dim/IR frames"),
    Config("I_crop_noupscale_noenhance", crop=(0, 0, 960, 1080), upscale=1.0, enhance="none",
           note="isolate upscale's own contribution: crop only, no enhance, no upscale"),
    Config("J_crop_noupscale_clahe_only", crop=(0, 0, 960, 1080), upscale=1.0, enhance="clahe_only",
           note="isolate upscale's own contribution: crop + CLAHE, no upscale"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=300.0)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset of config names to run, default = all")
    args = ap.parse_args()

    video = Path(args.video).expanduser()
    configs = CONFIGS if not args.configs else [c for c in CONFIGS if c.name in args.configs]

    out_dir = LAB_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for cfg in configs:
        print(f"--- {cfg.name} ({cfg.note}) ---")
        r = run_config(video, cfg, args.start, args.end)
        samples = r.pop("samples")
        GazeTrack(sortie=f"{video.stem}_{cfg.name}", source_video=str(video), seat="left",
                  sample_hz=cfg.hz, duration=args.end - args.start,
                  samples=samples).save(out_dir / f"{cfg.name}.json")
        print(f"  sampled={r['n_sampled']} detected={r['n_detected']} "
              f"coverage={r['coverage']:.1%} dropped_implausible={r['n_dropped_implausible']} "
              f"jitter_deg_median={r['jitter_deg_median']:.2f} wall={r['wall_s']}s")
        rows.append(r)

    with open(out_dir / "_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "coverage", "n_sampled", "n_detected",
                                          "n_dropped_implausible", "jitter_deg_median",
                                          "wall_s", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\ncomparison -> {out_dir / '_comparison.csv'}")

    best = max(rows, key=lambda r: r["coverage"])
    print(f"\nbest coverage: {best['config']} ({best['coverage']:.1%})")


if __name__ == "__main__":
    main()
