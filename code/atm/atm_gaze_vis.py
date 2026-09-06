"""Draw the cadet's gaze on the ATMOSPHERE-camera video.

Inputs:
  exports/atm/<sortie>/gaze_pfd.json   (atm_gaze_cluster.py -- gaze track + PFD label)
  exports/atm/<sortie>/aoi_boxes.json  (AOI rectangles on the ATM frame; PFD = primary)
  exports/atm/<sortie>/sync.json       (offset: atm_t = face_t + offset_b_minus_a_s)

Mapping (one anchor, per the crew's "inside gaze is ~95% PFD"):
  the PFD-cluster centroid is pinned to the PFD box centre; gaze angle is
  converted to pixels with a fixed deg->px scale from the ATM camera FOV
  (--hfov, default 90 for the wide "bigeyes" lens). This is a qualitative
  pointer -- the hard "on PFD %" comes from the classifier label, not the
  dot position.

  x_px = pfd_cx + sx*(yaw   - pfd_yaw)      (--flip-x to invert)
  y_px = pfd_cy - sy*(pitch - pfd_pitch)    (--flip-y to invert)

Overlay: fading gaze trail + dot (green on PFD, amber otherwise), AOI
boxes with the PFD box highlighted while on_pfd, and a readout with the
current state, a rolling 10 s on-PFD %, and the cumulative %.

    python3 code/atm/atm_gaze_vis.py --sortie "s1 c6" \\
        --video "/home/ilan/Documents/flight school/data/atm/s1/s1  c6 atm sorek bigeyes.mp4" \\
        --start 0 --duration 600
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
ATM_DIR = ROOT / "exports" / "atm"
FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"

LABEL_COL = {"on_pfd": (60, 200, 90), "other_inside": (235, 170, 40), "outside": (90, 160, 235)}


def _interp_track(samples):
    t = np.array([s["t"] for s in samples], float)
    yaw = np.array([s["yaw"] for s in samples], float)
    pitch = np.array([s["pitch"] for s in samples], float)
    lab = [s["label"] for s in samples]
    return t, yaw, pitch, lab


def _at(t_arr, y_arr, p_arr, lab, tt, max_gap=0.6):
    i = int(np.searchsorted(t_arr, tt))
    lo = max(0, i - 1)
    hi = min(len(t_arr) - 1, i)
    if abs(t_arr[lo] - tt) > max_gap and abs(t_arr[hi] - tt) > max_gap:
        return None
    j = lo if abs(t_arr[lo] - tt) <= abs(t_arr[hi] - tt) else hi
    if t_arr[hi] > t_arr[lo]:
        f = np.clip((tt - t_arr[lo]) / (t_arr[hi] - t_arr[lo]), 0, 1)
        yaw = (1 - f) * y_arr[lo] + f * y_arr[hi]
        pitch = (1 - f) * p_arr[lo] + f * p_arr[hi]
    else:
        yaw, pitch = y_arr[j], p_arr[j]
    return float(yaw), float(pitch), lab[j]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--video", required=True, help="the ATM-camera mp4")
    ap.add_argument("--start", type=float, default=0.0, help="ATM-clock start (s)")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--hfov", type=float, default=90.0, help="ATM cam horizontal FOV (deg)")
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = ATM_DIR / args.sortie
    gz = json.loads((d / "gaze_pfd.json").read_text())
    aoi = json.loads((d / "aoi_boxes.json").read_text())
    sync = json.loads((d / "sync.json").read_text())
    offset = float(sync["offset_b_minus_a_s"])        # atm_t = face_t + offset

    t_arr, y_arr, p_arr, lab = _interp_track(gz["samples"])
    cen_y, cen_p = gz["pfd_centroid"]["yaw"], gz["pfd_centroid"]["pitch"]

    boxes = aoi["boxes"]
    pfd_box = boxes[aoi.get("primary", "PFD")]
    pfd_cx = (pfd_box[0] + pfd_box[2]) / 2
    pfd_cy = (pfd_box[1] + pfd_box[3]) / 2
    fw, fh = aoi["frame_size"]
    sx = (fw / args.hfov) * (-1 if args.flip_x else 1)
    sy = (fw / args.hfov) * (-1 if args.flip_y else 1)   # square px; use hfov scale on both axes

    # CFR window so frames <-> audio stay locked
    probe = cv2.VideoCapture(args.video)
    raw_fps = probe.get(cv2.CAP_PROP_FPS)
    probe.release()
    if not raw_fps or raw_fps <= 0 or raw_fps > 240:
        sys.exit(f"bad fps {raw_fps}")
    fps = round(raw_fps, 3)
    cfr = d / f".cfr_{int(args.start)}_{int(args.duration)}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(args.start), "-t", str(args.duration),
                    "-i", args.video, "-r", f"{fps}", "-vsync", "cfr",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-ar", "48000", str(cfr)], check=True)

    cap = cv2.VideoCapture(str(cfr))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ax, ay = w / fw, h / fh                     # aoi coords -> actual frame scale
    font = ImageFont.truetype(FONT_PATH, max(22, h // 34))
    small = ImageFont.truetype(FONT_PATH, max(16, h // 48))

    out_path = Path(args.out or d / f"gaze_atm_{int(args.start)}_{int(args.duration)}.mp4")
    silent = out_path.with_suffix(".silent.mp4")
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trail = []                                  # (x, y, label) recent points
    roll = []                                   # (t, is_pfd) last 10 s
    cum_pfd = cum_tot = 0

    for i in tqdm(range(n_frames), desc="gaze->atm", unit="f"):
        ok, frame = cap.read()
        if not ok:
            break
        t_atm = args.start + i / fps
        t_face = t_atm - offset                 # face_t = atm_t - offset
        got = _at(t_arr, y_arr, p_arr, lab, t_face)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, "RGBA")

        cur_lab = got[2] if got else None
        # AOI boxes
        for name, b in boxes.items():
            x0, y0, x1, y1 = b[0] * ax, b[1] * ay, b[2] * ax, b[3] * ay
            hot = (name == aoi.get("primary", "PFD") and cur_lab == "on_pfd")
            col = (60, 200, 90, 255) if hot else (200, 200, 200, 150)
            draw.rectangle([x0, y0, x1, y1], outline=col, width=4 if hot else 2)
            draw.text((x0 + 4, y0 + 2), name, font=small, fill=col)

        if got:
            yaw, pitch, lb = got
            gx = pfd_cx + sx * (yaw - cen_y)
            gy = pfd_cy - sy * (pitch - cen_p)
            gx, gy = gx * ax, gy * ay
            off = not (0 <= gx <= w and 0 <= gy <= h)
            gx = float(np.clip(gx, 14, w - 14))          # keep it on-screen; edge = "off that way"
            gy = float(np.clip(gy, 14, h - 14))
            trail.append((gx, gy, lb))
            trail[:] = trail[-18:]
            for k, (tx, ty, tl) in enumerate(trail[:-1]):
                a = int(60 * (k + 1) / len(trail))
                draw.line([tx, ty, trail[k + 1][0], trail[k + 1][1]],
                          fill=LABEL_COL.get(tl, (200, 200, 200)) + (a,), width=3)
            c = LABEL_COL.get(lb, (200, 200, 200))
            r = 22 if off else 16                        # bigger hollow ring when clamped to an edge
            draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline=c + (255,), width=4)
            if not off:
                draw.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], fill=c + (255,))
            is_pfd = lb == "on_pfd"
        else:
            is_pfd = False

        roll.append((t_atm, is_pfd))
        roll[:] = [(rt, rp) for rt, rp in roll if rt >= t_atm - 10]
        roll_pct = 100 * sum(rp for _, rp in roll) / max(len(roll), 1)
        if got:
            cum_tot += 1
            cum_pfd += int(is_pfd)
        cum_pct = 100 * cum_pfd / max(cum_tot, 1)

        state = {"on_pfd": "ON PFD", "other_inside": "inside (not PFD)",
                 "outside": "OUTSIDE", None: "no gaze"}[cur_lab]
        sc = LABEL_COL.get(cur_lab, (150, 150, 150))
        y0 = 40                                          # clear the review-tool's own header
        draw.rectangle([0, y0, 430, y0 + 96], fill=(0, 0, 0, 170))
        draw.text((14, y0 + 6), state, font=font, fill=sc + (255,))
        draw.text((14, y0 + 6 + font.size + 6),
                  f"PFD  10s {roll_pct:4.0f}%   all {cum_pct:4.0f}%",
                  font=small, fill=(235, 235, 235, 255))
        draw.text((14, y0 + 6 + font.size + 6 + small.size + 4),
                  f"t={t_atm:6.1f}s  (face {t_face:6.1f}s)", font=small, fill=(180, 180, 180, 255))

        vw.write(cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR))

    vw.release()
    cap.release()
    print("muxing audio ...")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(cfr),
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
                    "-crf", "20", "-c:a", "aac", "-shortest", str(out_path)], check=True)
    silent.unlink(missing_ok=True)
    cfr.unlink(missing_ok=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
