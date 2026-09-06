"""Side-by-side: face cam (left) + ATM cam (right), time-synced, with the
cadet's gaze drawn on BOTH.

  left  (face cam)  -- gaze arrow from the cadet's head + ON-PFD/OUTSIDE label
  right (ATM cam)   -- gaze dot + trail + AOI boxes + rolling / cumulative
                       on-PFD %   (same as atm_gaze_vis.py)

Sync: both are cut to constant fps; the face clip is started
`offset` seconds ahead (from sync.json: atm_t = face_t + offset, offset
< 0), so face frame i lines up with ATM frame i with no per-frame seek.

    python3 code/atm/atm_gaze_pair.py --sortie "s1 c6" \\
        --face "/home/ilan/Documents/flight school/data/s1 c6.mp4" \\
        --atm  "/home/ilan/Documents/flight school/data/atm/s1/s1  c6 atm sorek bigeyes.mp4" \\
        --start 300 --duration 300
"""

import argparse
import json
import subprocess
import sys
from math import sin, radians
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
ATM_DIR = ROOT / "exports" / "atm"
FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
LABEL_COL = {"on_pfd": (60, 200, 90), "other_inside": (235, 170, 40), "outside": (90, 160, 235)}
FACE_CROP_CENTER = (592, 455)   # centre of the extract_gaze --crop 224,40,960,870 region
FPS = 25
HALF_W, HALF_H = 1280, 720


def _track(samples):
    t = np.array([s["t"] for s in samples], float)
    return (t, np.array([s["yaw"] for s in samples], float),
            np.array([s["pitch"] for s in samples], float), [s["label"] for s in samples])


def _at(t, y, p, lab, tt, gap=0.6):
    i = int(np.searchsorted(t, tt))
    lo, hi = max(0, i - 1), min(len(t) - 1, i)
    if min(abs(t[lo] - tt), abs(t[hi] - tt)) > gap:
        return None
    if t[hi] > t[lo]:
        f = float(np.clip((tt - t[lo]) / (t[hi] - t[lo]), 0, 1))
        return ((1 - f) * y[lo] + f * y[hi], (1 - f) * p[lo] + f * p[hi],
                lab[lo] if f < 0.5 else lab[hi])
    return float(y[lo]), float(p[lo]), lab[lo]


def _cfr(video, start, dur, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start}", "-t", f"{dur}",
                    "-i", str(video), "-r", str(FPS), "-vsync", "cfr", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", str(out)], check=True)


def _fit(frame):
    h, w = frame.shape[:2]
    s = min(HALF_W / w, HALF_H / h)
    r = cv2.resize(frame, (int(w * s), int(h * s)))
    canvas = np.zeros((HALF_H, HALF_W, 3), np.uint8)
    y0, x0 = (HALF_H - r.shape[0]) // 2, (HALF_W - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--atm", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="ATM-clock start (s)")
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = ATM_DIR / args.sortie
    gz = json.loads((d / "gaze_pfd.json").read_text())
    aoi = json.loads((d / "aoi_boxes.json").read_text())
    offset = float(json.loads((d / "sync.json").read_text())["offset_b_minus_a_s"])

    t_arr, y_arr, p_arr, lab = _track(gz["samples"])
    cen_y, cen_p = gz["pfd_centroid"]["yaw"], gz["pfd_centroid"]["pitch"]
    boxes = aoi["boxes"]
    pb = boxes[aoi.get("primary", "PFD")]
    pfd_cx, pfd_cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
    fw, fh = aoi["frame_size"]
    sx = (fw / args.hfov) * (-1 if args.flip_x else 1)
    sy = (fw / args.hfov) * (-1 if args.flip_y else 1)

    tmp = d / ".pair"
    tmp.mkdir(exist_ok=True)
    face_cfr, atm_cfr = tmp / "face.mp4", tmp / "atm.mp4"
    face_start = args.start - offset            # offset<0 -> face starts later? atm=face+offset
    print(f"cutting CFR clips (face start {face_start:.2f}s, atm start {args.start:.2f}s) ...")
    _cfr(args.face, face_start, args.duration, face_cfr)
    _cfr(args.atm, args.start, args.duration, atm_cfr)

    cf, ca = cv2.VideoCapture(str(face_cfr)), cv2.VideoCapture(str(atm_cfr))
    n = min(int(cf.get(cv2.CAP_PROP_FRAME_COUNT)), int(ca.get(cv2.CAP_PROP_FRAME_COUNT)))
    font = ImageFont.truetype(FONT_PATH, 30)
    small = ImageFont.truetype(FONT_PATH, 22)

    out_path = Path(args.out or d / f"gaze_pair_{int(args.start)}_{int(args.duration)}.mp4")
    silent = out_path.with_suffix(".silent.mp4")
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (HALF_W * 2, HALF_H))

    trail, roll = [], []
    cum_pfd = cum_tot = 0
    for i in tqdm(range(n), desc="pair", unit="f"):
        okf, ff = cf.read()
        oka, fa = ca.read()
        if not (okf and oka):
            break
        t_atm = args.start + i / FPS
        t_face = t_atm - offset
        got = _at(t_arr, y_arr, p_arr, lab, t_face)
        cur = got[2] if got else None
        col = LABEL_COL.get(cur, (170, 170, 170))

        # ---- face (left) ----
        fim = Image.fromarray(cv2.cvtColor(ff, cv2.COLOR_BGR2RGB))
        fd = ImageDraw.Draw(fim, "RGBA")
        ax0, ay0 = FACE_CROP_CENTER
        if got:
            yaw, pitch, _ = got
            ex = ax0 + 9 * (yaw - cen_y) * (-1 if args.flip_x else 1)
            ey = ay0 - 9 * (pitch - cen_p)
            fd.line([ax0, ay0, ex, ey], fill=col + (255,), width=6)
            fd.ellipse([ex - 12, ey - 12, ex + 12, ey + 12], outline=col + (255,), width=5)
        fd.ellipse([ax0 - 7, ay0 - 7, ax0 + 7, ay0 + 7], fill=col + (255,))
        fd.rectangle([0, 44, 360, 104], fill=(0, 0, 0, 170))
        fd.text((12, 50), {"on_pfd": "CADET → PFD", "other_inside": "CADET inside",
                           "outside": "CADET → outside", None: "no gaze"}[cur],
                font=font, fill=col + (255,))

        # ---- ATM (right) ----
        aim = Image.fromarray(cv2.cvtColor(fa, cv2.COLOR_BGR2RGB))
        adr = ImageDraw.Draw(aim, "RGBA")
        axs, ays = fa.shape[1] / fw, fa.shape[0] / fh
        for name, b in boxes.items():
            hot = (name == aoi.get("primary", "PFD") and cur == "on_pfd")
            c = (60, 200, 90, 255) if hot else (200, 200, 200, 150)
            adr.rectangle([b[0] * axs, b[1] * ays, b[2] * axs, b[3] * ays],
                          outline=c, width=4 if hot else 2)
            adr.text((b[0] * axs + 4, b[1] * ays + 2), name, font=small, fill=c)
        if got:
            yaw, pitch, lb = got
            gx = (pfd_cx + sx * (yaw - cen_y)) * axs
            gy = (pfd_cy - sy * (pitch - cen_p)) * ays
            offscr = not (0 <= gx <= fa.shape[1] and 0 <= gy <= fa.shape[0])
            gx = float(np.clip(gx, 14, fa.shape[1] - 14))
            gy = float(np.clip(gy, 14, fa.shape[0] - 14))
            trail.append((gx, gy, lb))
            trail[:] = trail[-18:]
            for k in range(len(trail) - 1):
                a = int(60 * (k + 1) / len(trail))
                adr.line([*trail[k][:2], *trail[k + 1][:2]],
                         fill=LABEL_COL.get(trail[k][2], (200, 200, 200)) + (a,), width=3)
            r = 22 if offscr else 16
            adr.ellipse([gx - r, gy - r, gx + r, gy + r], outline=col + (255,), width=4)
            if not offscr:
                adr.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], fill=col + (255,))
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
        adr.rectangle([0, 44, 430, 140], fill=(0, 0, 0, 170))
        adr.text((12, 50), {"on_pfd": "ON PFD", "other_inside": "inside (not PFD)",
                            "outside": "OUTSIDE", None: "no gaze"}[cur],
                 font=font, fill=col + (255,))
        adr.text((12, 90), f"PFD  10s {roll_pct:4.0f}%   all {cum_pct:4.0f}%",
                 font=small, fill=(235, 235, 235, 255))
        adr.text((12, 114), f"t={t_atm:6.1f}s  (face {t_face:6.1f}s)",
                 font=small, fill=(180, 180, 180, 255))

        left = _fit(cv2.cvtColor(np.array(fim.convert("RGB")), cv2.COLOR_RGB2BGR))
        right = _fit(cv2.cvtColor(np.array(aim.convert("RGB")), cv2.COLOR_RGB2BGR))
        vw.write(np.hstack([left, right]))

    vw.release()
    cf.release()
    ca.release()
    print("muxing ATM audio ...")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent),
                    "-ss", str(args.start), "-t", str(args.duration), "-i", str(args.atm),
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
                    "-crf", "21", "-c:a", "aac", "-shortest", str(out_path)], check=True)
    silent.unlink(missing_ok=True)
    for f in (face_cfr, atm_cfr):
        f.unlink(missing_ok=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
