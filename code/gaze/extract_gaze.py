"""Extract an approximate gaze track from a cabin-facing sortie video
(mediapipe FaceLandmarker: head pose + iris position), one GazeTrack per
seat (left / right crew position).

Method, per sampled frame (--hz, default 5 -- fixations run >100ms so this
is plenty, and it keeps a 45min video tractable):
  1. white-balance (gray-world) + CLAHE contrast enhancement -- this
     footage is dim and has a strong green cast (night/IR-lit cabin) that
     the stock face detector otherwise misses almost every frame.
  2. mediapipe FaceLandmarker (up to 2 faces), with the facial
     transformation matrix enabled for head pose.
  3. head yaw/pitch/roll from that matrix's rotation part.
  4. eye-in-head yaw/pitch from the iris landmarks' offset within each eye's
     corners, scaled by an assumed +/-20 deg comfortable eye-in-head range.
  5. gaze = head pose + eye-in-head offset (a standard cheap appearance-based
     proxy -- not metrically calibrated, but consistent within a sortie).
  6. seat = "left"/"right" by which half of the frame the face center falls in.

Writes exports/gaze/<sortie>/<seat>.json (one GazeTrack per seat found).

    python3 code/gaze/extract_gaze.py "/path/s1 c2.mp4"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import GazeSample, GazeTrack  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
GAZE_DIR = ROOT / "exports" / "gaze"
MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"

EYE_RANGE_DEG = 20.0   # assumed comfortable eye-in-head excursion -> degrees scale factor
MAX_PLAUSIBLE_DEG = 85.0  # a detected (mostly-frontal) face can't really be turned further
                          # than this; bigger yaw/pitch is a bad rotation-matrix decomposition
                          # on that frame (occlusion/motion blur), not real head motion -- drop it
# mediapipe 478-landmark indices (iris refinement on)
R_EYE = {"outer": 33, "inner": 133, "top": 159, "bot": 145, "iris": 468}
L_EYE = {"outer": 263, "inner": 362, "top": 386, "bot": 374, "iris": 473}


def _enhance(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    mean = f.reshape(-1, 3).mean(0)
    f = f * (mean.mean() / (mean + 1e-6))
    frame = np.clip(f, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _head_pose(matrix) -> tuple[float, float, float]:
    R = np.array(matrix).reshape(4, 4)[:3, :3]
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(yaw), float(pitch), float(roll)


def _eye_offset(lm, w, h, eye) -> tuple[float, float] | None:
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


def extract(video: str | Path, hz: float = 5.0, num_faces: int = 2,
            crop: tuple[int, int, int, int] | None = None, upscale: float = 1.0,
            seat_override: str | None = None) -> list[GazeTrack]:
    """crop = (x0,y0,x1,y1) px applied to every frame before detection
    (use it to isolate one seat and drop lens flare / tool chrome); upscale
    enlarges that crop before mediapipe (small/soft faces detect better);
    seat_override forces the output seat label (with a crop, left/right by
    face x-position is meaningless)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = vision.FaceLandmarkerOptions(
        base_options=base, num_faces=num_faces, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE)
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps else 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    step = max(1, int(round(fps / hz)))

    from collections import defaultdict
    seats: dict[str, list[GazeSample]] = defaultdict(list)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            if crop is not None:
                x0, y0, x1, y1 = crop
                frame = frame[max(0, y0):y1, max(0, x0):x1]
            if upscale and upscale != 1.0:
                frame = cv2.resize(frame, None, fx=upscale, fy=upscale,
                                   interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(_enhance(frame), cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            for lm, matrix in zip(res.face_landmarks, res.facial_transformation_matrixes):
                if seat_override:
                    seat = seat_override
                else:
                    cx = float(np.mean([p.x for p in lm])) * frame.shape[1]
                    seat = "left" if cx < w / 2 else "right"
                hyaw, hpitch, _ = _head_pose(matrix)
                offs = [o for o in (_eye_offset(lm, frame.shape[1], frame.shape[0], R_EYE),
                                    _eye_offset(lm, frame.shape[1], frame.shape[0], L_EYE)) if o]
                if offs:
                    dx = float(np.mean([o[0] for o in offs])) * EYE_RANGE_DEG
                    dy = float(np.mean([o[1] for o in offs])) * EYE_RANGE_DEG
                else:
                    dx = dy = 0.0
                yaw, pitch = hyaw + dx, hpitch + dy
                if abs(yaw) > MAX_PLAUSIBLE_DEG or abs(pitch) > MAX_PLAUSIBLE_DEG:
                    continue   # bad frame (occlusion/blur threw the pose off) -- drop, don't fabricate
                seats[seat].append(GazeSample(
                    t=round(t, 3), seat=seat, yaw=round(yaw, 2),
                    pitch=round(pitch, 2), head_yaw=round(hyaw, 2),
                    head_pitch=round(hpitch, 2), conf=1.0))
        idx += 1
    cap.release()

    tracks = []
    for seat, samples in seats.items():
        if samples:
            tracks.append(GazeTrack(sortie=video.stem, source_video=str(video), seat=seat,
                                    sample_hz=hz, duration=duration, samples=samples))
    return tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 px -- isolate one seat")
    ap.add_argument("--upscale", type=float, default=1.0)
    ap.add_argument("--seat", default=None, help="force seat label (use with --crop)")
    args = ap.parse_args()

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    nf = 1 if crop else 2
    tracks = extract(args.video, hz=args.hz, num_faces=nf, crop=crop,
                     upscale=args.upscale, seat_override=args.seat)
    out_dir = GAZE_DIR / Path(args.video).stem
    for tr in tracks:
        out = out_dir / f"{tr.seat}.json"
        tr.save(out)
        cov = len(tr.samples) / max(tr.duration * args.hz, 1)
        print(f"{tr.seat}: {len(tr.samples)} samples, coverage {cov:.1%} of frames sampled -> {out}")
    if not tracks:
        print("no faces detected at all -- check video / lighting")


if __name__ == "__main__":
    main()
