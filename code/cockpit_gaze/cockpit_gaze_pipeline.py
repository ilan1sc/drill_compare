"""Cadet gaze, extracted from the copilot-cam face video (06), projected
onto the pilot-cam cockpit video (02) so you can see WHERE on the panel
the cadet was actually looking -- and debug the gaze algorithm itself
(face features / angles) side by side with where that lands.

Inputs, from one cadet session folder under .../26.1.s1.extrct_v2/:
  06_F2-COPILOT CAM COCKPIT- (...).mp4   -- face-on camera. Cadet is the
                                             LEFT-seat face in this view.
  02_F2-PILOT CAM COCKPIT- (...).mp4     -- rear/over-the-shoulder cockpit
                                             view. The two Garmin screens
                                             are visible here; PFD = the
                                             left one (pfd_finder.py).

Camera clock sync: both cameras are exported from the same recording
session and manifest.json carries each stream's wall-clock start
(`start_unix`) -- no OCR-based sync needed, just:
    t02 = t06 + (start_unix[06] - start_unix[02])

Gaze -> pixel projection ("homography"): NOT a full 4-point planar
homography (we don't have correspondences between the two camera
planes -- one watches a face, the other watches a panel from behind).
Instead, the same "one physical anchor + angular scale" model already
used for the ATM camera (code/atm/atm_gaze_cluster.py +
atm_gaze_vis.py): the auto-detected PFD-looking gaze cluster centroid
(yaw0, pitch0) is pinned to the PFD box's pixel centre (px0, py0), and
degrees are converted to pixels via the 02 camera's (assumed) FOV:
    x = px0 - flip_x * sx * (yaw - yaw0)
    y = py0 - sy * (pitch - pitch0)
sx = frame_w / hfov_deg, sy = frame_h / vfov_deg. flip_x = -1 by
default: the copilot cam (06, faces the crew) and the pilot cam (02,
faces the panel from behind the crew) are mirrored relative to each
other, so a cadet gaze-yaw sign in one is reversed in the other
(--no-flip-x to disable if this looks wrong for a given session).

Outputs, under exports/cockpit_gaze/<cadet label>/:
  calibration.json      -- PFD box, anchor, scale, flip -- the "homography"
  gaze_track.json        -- raw extract_gaze GazeTrack (06, left seat)
  gaze_pfd.json           -- clustered on_pfd/other_inside/outside labels
  per_frame.csv           -- t, x, y, state, yaw, pitch, conf   (one row/sample)
  face_cam_augmented.mp4  -- 06, cropped to left seat, with face box +
                              yaw/pitch + state overlay (debug view)
  cockpit_augmented.mp4   -- 02, with PFD red box + projected gaze dot/trail
  heatmap.png             -- static 02 frame + gaze-density heatmap overlay

    python3 code/cockpit_gaze/cockpit_gaze_pipeline.py \\
        --cadet-dir ".../26.1.s1.extrct_v2/26.1.s1.c6.sorek_bigeyes_exp_v2" \\
        --start 0 --duration 300
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gaze"))
sys.path.insert(0, str(ROOT / "code" / "atm"))
sys.path.insert(0, str(ROOT / "gazelab2"))
from schema import GazeTrack  # noqa: E402
from atm_gaze_cluster import cluster  # noqa: E402
from lab import Config, run_config  # noqa: E402
from pfd_finder import find_pfd_stable  # noqa: E402

EXPORT_DIR = ROOT / "exports" / "cockpit_gaze"
LEFT_CROP = (0, 0, 960, 1080)   # left half of the 06 (1920x1080) copilot cam


def _find_stream(cadet_dir: Path, prefix: str) -> Path:
    hits = list(cadet_dir.glob(f"{prefix}*.mp4"))
    if not hits:
        raise SystemExit(f"no {prefix}*.mp4 under {cadet_dir}")
    return hits[0]


def _manifest_offset(cadet_dir: Path, file06: Path, file02: Path) -> float:
    m = json.loads((cadet_dir / "manifest.json").read_text())
    starts = {s["file"]: s["start_unix"] for s in m["streams"] if s["type"] == "camera"}
    return starts[file06.name] - starts[file02.name]


def build_track(face_video: Path, hz: float, start: float, end: float) -> GazeTrack:
    cfg = Config("cockpit", crop=LEFT_CROP, upscale=1.0, enhance="none", hz=hz)
    r = run_config(face_video, cfg, start, end)
    return GazeTrack(sortie=face_video.stem, source_video=str(face_video), seat="left",
                     sample_hz=hz, duration=end - start, samples=r["samples"])


def calibrate(cockpit_video: Path, cluster_res: dict, hfov: float, vfov: float, flip_x: bool) -> dict:
    box = find_pfd_stable(cockpit_video)
    if box is None:
        raise SystemExit("pfd_finder found nothing in the cockpit video")
    cap = cv2.VideoCapture(str(cockpit_video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    x0, y0, x1, y1 = box
    return {
        "cockpit_video": str(cockpit_video), "frame_w": w, "frame_h": h,
        "pfd_box_xyxy": list(box), "pfd_center_px": [(x0 + x1) / 2, (y0 + y1) / 2],
        "pfd_anchor_deg": cluster_res["pfd_centroid"],
        "hfov_deg": hfov, "vfov_deg": vfov, "flip_x": flip_x,
        "sx_px_per_deg": w / hfov, "sy_px_per_deg": h / vfov,
    }


def project(yaw: float, pitch: float, calib: dict) -> tuple[float, float]:
    cx, cy = calib["pfd_center_px"]
    y0, p0 = calib["pfd_anchor_deg"]["yaw"], calib["pfd_anchor_deg"]["pitch"]
    sx, sy = calib["sx_px_per_deg"], calib["sy_px_per_deg"]
    fx = -1.0 if calib["flip_x"] else 1.0
    x = cx + fx * sx * (yaw - y0)
    y = cy - sy * (pitch - p0)
    return x, y


LABEL_COL = {"on_pfd": (60, 200, 90), "other_inside": (235, 170, 40), "outside": (90, 160, 235)}


def render_face_cam(face_video: Path, gaze_res: dict, out_path: Path, start: float, end: float, hz: float):
    samples = gaze_res["samples"]
    ts = np.array([s["t"] for s in samples])
    cap = cv2.VideoCapture(str(face_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    x0c, y0c, x1c, y1c = LEFT_CROP
    dw, dh = 640, int(640 * (y1c - y0c) / (x1c - x0c))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (dw, dh))
    n = int((end - start) * fps)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t = start + i / fps
        frame = frame[y0c:y1c, x0c:x1c]
        disp = cv2.resize(frame, (dw, dh))
        j = int(np.searchsorted(ts, t))
        j = min(max(j, 0), len(samples) - 1) if samples else -1
        if j >= 0 and abs(ts[j] - t) < 1.0 / hz * 2:
            s = samples[j]
            col = LABEL_COL.get(s["label"], (200, 200, 200))
            cv2.putText(disp, f"yaw {s['yaw']:5.1f} pitch {s['pitch']:5.1f}  {s['label']}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            cx, cy = dw // 2, dh // 2
            ex = int(cx + 3 * s["yaw"])
            ey = int(cy - 3 * s["pitch"])
            cv2.arrowedLine(disp, (cx, cy), (ex, ey), col, 3, tipLength=0.25)
        else:
            cv2.putText(disp, "NO DETECTION", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 220), 2)
        writer.write(disp)
    writer.release()
    cap.release()
    print("->", out_path)


def render_cockpit_cam(cockpit_video: Path, per_frame: list[dict], calib: dict,
                        out_path: Path, start: float, end: float):
    cap = cv2.VideoCapture(str(cockpit_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    w, h = calib["frame_w"], calib["frame_h"]
    scale = 1280 / w
    dw, dh = int(w * scale), int(h * scale)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (dw, dh))
    x0, y0, x1, y1 = [int(v * scale) for v in calib["pfd_box_xyxy"]]
    ts02 = np.array([r["t02"] for r in per_frame])
    trail = []
    n = int((end - start) * fps)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t = start + i / fps
        disp = cv2.resize(frame, (dw, dh))
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.putText(disp, "PFD", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        j = int(np.searchsorted(ts02, t))
        j = min(max(j, 0), len(per_frame) - 1) if per_frame else -1
        if j >= 0 and per_frame[j]["state"] is not None:
            r = per_frame[j]
            px, py = int(r["x"] * scale), int(r["y"] * scale)
            trail.append((px, py, r["state"]))
            trail[:] = trail[-15:]
            for k, (tx, ty, tl) in enumerate(trail):
                alpha = (k + 1) / len(trail)
                col = LABEL_COL.get(tl, (200, 200, 200))
                cv2.circle(disp, (tx, ty), int(4 + 6 * alpha), col, -1)
            cv2.putText(disp, r["state"], (8, dh - 15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, LABEL_COL.get(r["state"], (200, 200, 200)), 2)
        writer.write(disp)
    writer.release()
    cap.release()
    print("->", out_path)


def make_heatmap(cockpit_video: Path, per_frame: list[dict], calib: dict, out_path: Path):
    cap = cv2.VideoCapture(str(cockpit_video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    h, w = frame.shape[:2]
    heat = np.zeros((h, w), np.float32)
    for r in per_frame:
        if r["state"] is None:
            continue
        x, y = int(r["x"]), int(r["y"])
        if 0 <= x < w and 0 <= y < h:
            heat[y, x] += 1.0
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=25)
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    mask = (heat > 0.02)[:, :, None]
    out = frame.copy()
    out = np.where(mask, cv2.addWeighted(frame, 0.45, heat_color, 0.55, 0), out)
    x0, y0, x1, y1 = calib["pfd_box_xyxy"]
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 4)
    cv2.imwrite(str(out_path), out)
    print("->", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadet-dir", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--hfov", type=float, default=90.0, help="02 camera horizontal FOV, degrees")
    ap.add_argument("--vfov", type=float, default=70.0, help="02 camera vertical FOV, degrees")
    ap.add_argument("--no-flip-x", action="store_true",
                    help="disable the yaw-sign flip between 06 and 02 (mirrored camera positions)")
    args = ap.parse_args()

    cadet_dir = Path(args.cadet_dir)
    face_video = _find_stream(cadet_dir, "06_F2-COPILOT")
    cockpit_video = _find_stream(cadet_dir, "02_F2-PILOT")
    label = cadet_dir.name.replace("_exp_v2", "")
    out_dir = EXPORT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    offset = _manifest_offset(cadet_dir, face_video, cockpit_video)
    print(f"t02 = t06 + {offset:.3f}s  (from manifest.json start_unix)")

    print("extracting gaze from", face_video.name)
    track = build_track(face_video, args.hz, args.start, args.start + args.duration)
    track.save(out_dir / "gaze_track.json")
    print(f"  {len(track.samples)} samples "
          f"({len(track.samples) / max((args.duration) * args.hz, 1):.1%} coverage)")

    print("clustering on_pfd / other_inside / outside")
    res = cluster(track)
    (out_dir / "gaze_pfd.json").write_text(json.dumps(res, indent=2))
    print(f"  PFD anchor yaw={res['pfd_centroid']['yaw']} pitch={res['pfd_centroid']['pitch']}"
          f"  on_pfd={res['on_pfd_pct']}% other_inside={res['other_inside_pct']}%"
          f" outside={res['outside_pct']}%")

    print("locating PFD in", cockpit_video.name)
    calib = calibrate(cockpit_video, res, args.hfov, args.vfov, flip_x=not args.no_flip_x)
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
    print(f"  PFD box {calib['pfd_box_xyxy']}  flip_x={calib['flip_x']}")

    per_frame = []
    for s in res["samples"]:
        x, y = project(s["yaw"], s["pitch"], calib)
        per_frame.append({"t06": s["t"], "t02": s["t"] + offset, "x": x, "y": y,
                          "state": s["label"], "yaw": s["yaw"], "pitch": s["pitch"]})
    with open(out_dir / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["t06", "t02", "x", "y", "state", "yaw", "pitch"])
        w.writeheader()
        w.writerows(per_frame)
    print(f"-> {out_dir / 'per_frame.csv'} ({len(per_frame)} rows)")

    render_face_cam(face_video, res, out_dir / "face_cam_augmented.mp4",
                    args.start, args.start + args.duration, args.hz)
    render_cockpit_cam(cockpit_video, per_frame, calib, out_dir / "cockpit_augmented.mp4",
                       args.start + offset, args.start + offset + args.duration)
    make_heatmap(cockpit_video, per_frame, calib, out_dir / "heatmap.png")

    print(f"\nall outputs -> {out_dir}")


if __name__ == "__main__":
    main()
