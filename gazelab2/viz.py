"""Render a side-by-side-able comparison video: baseline (full frame,
both seats) vs the winning fix (crop to left seat, no upscale, no
enhance) -- so the coverage jump (24.7% -> 98%) is visible, not just a
number.

Draws, per native frame: a green box + yaw/pitch readout when a face is
detected on that config's last processed sample, a red "NO DETECTION"
banner otherwise (held until the next sampled frame, same 5 Hz cadence
lab.py used).

    venv/bin/python viz.py --video "<path>/s1 c6.mp4" --start 0 --end 60
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR.parent / "code" / "gaze"))

from lab import Config, MAX_PLAUSIBLE_DEG, R_EYE, L_EYE, _enhance, _head_pose, _eye_offset  # noqa: E402

BASELINE = Config("A_baseline", crop=None, upscale=1.0, enhance="graygrad_clahe")
FIXED = Config("I_fixed", crop=(0, 0, 960, 1080), upscale=1.0, enhance="none")


def render(video: Path, cfg: Config, out_path: Path, start: float, end: float, label: str):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base = mp_python.BaseOptions(model_asset_path=str(LAB_DIR.parent / "code" / "gaze" / "face_landmarker.task"))
    opts = vision.FaceLandmarkerOptions(
        base_options=base, num_faces=1, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=cfg.min_det_conf, min_face_presence_confidence=cfg.min_pres_conf,
        running_mode=vision.RunningMode.IMAGE)
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / cfg.hz)))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    out_w, out_h = (960, 1080) if cfg.crop else (1920, 1080)
    disp_scale = 640 / out_w
    dw, dh = int(out_w * disp_scale), int(out_h * disp_scale)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (dw, dh))

    last_detected = False
    last_yaw = last_pitch = 0.0
    n_frames = int((end - start) * fps)
    idx = 0
    while idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if cfg.crop is not None:
            x0, y0, x1, y1 = cfg.crop
            frame = frame[max(0, y0):y1, max(0, x0):x1]
        if idx % step == 0:
            rgb = cv2.cvtColor(_enhance(frame, cfg.enhance, cfg.clahe_clip), cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                matrix = res.facial_transformation_matrixes[0]
                hyaw, hpitch, _ = _head_pose(matrix)
                offs = [o for o in (_eye_offset(lm, frame.shape[1], frame.shape[0], R_EYE),
                                    _eye_offset(lm, frame.shape[1], frame.shape[0], L_EYE)) if o]
                dx = float(np.mean([o[0] for o in offs])) * cfg.eye_range_deg if offs else 0.0
                dy = float(np.mean([o[1] for o in offs])) * cfg.eye_range_deg if offs else 0.0
                yaw, pitch = hyaw + dx, hpitch + dy
                last_detected = abs(yaw) <= MAX_PLAUSIBLE_DEG and abs(pitch) <= MAX_PLAUSIBLE_DEG
                last_yaw, last_pitch = yaw, pitch
                xs = [p.x * frame.shape[1] for p in lm]
                ys = [p.y * frame.shape[0] for p in lm]
                last_box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            else:
                last_detected = False

        disp = cv2.resize(frame, (dw, dh))
        if last_detected:
            bx0, by0, bx1, by1 = [int(v * disp_scale) for v in last_box]
            cv2.rectangle(disp, (bx0, by0), (bx1, by1), (60, 210, 90), 2)
            cv2.putText(disp, f"yaw {last_yaw:5.1f}  pitch {last_pitch:5.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 210, 90), 2)
        else:
            cv2.putText(disp, "NO DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 220), 2)
        cv2.putText(disp, label, (10, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        writer.write(disp)
        idx += 1
    writer.release()
    cap.release()
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=60.0)
    args = ap.parse_args()
    video = Path(args.video).expanduser()
    out_dir = LAB_DIR / "videos"
    out_dir.mkdir(exist_ok=True)
    render(video, BASELINE, out_dir / "baseline_full_frame.mp4", args.start, args.end,
           "A: baseline (full frame, both seats)")
    render(video, FIXED, out_dir / "fixed_left_crop.mp4", args.start, args.end,
           "B: fixed (left-seat crop, no upscale/enhance)")


if __name__ == "__main__":
    main()
