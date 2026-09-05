"""Sync the multi-camera videos by OCR-ing the wall-clock burned into the
top-left of every one of them (the review-tool header:
"F2-... COCKPIT- (ip) - DD/MM/YYYY HH:MM:SS.mmm").

Each video (face cam, ATM cam, PFD screen-recording) is an independent
screen recording with its own arbitrary t=0, but they all stamp the SAME
wall clock, so:

    offset(A -> B) = median over sampled frames of
                     (wallclock_B(frame at t) - wallclock_A(frame at t))

    ->  B_time = A_time + offset

Reuses rapidocr (already installed for code/pfd.py). The header is large
high-contrast text so a handful of frames is plenty.

    python3 code/atm/sync_ocr.py \\
        --a "/path/s1 c6.mp4" --b "/path/atm/s1/s1  c6 atm sorek bigeyes.mp4" \\
        --sortie "s1 c6"
-> exports/atm/<sortie>/sync.json  { a, b, offset_b_minus_a_s, samples }
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ATM_DIR = ROOT / "exports" / "atm"
HEADER_CROP = (0.0, 0.0, 0.60, 0.045)   # x0,y0,x1,y1 fractions -- top-left header strip
TS_RE = re.compile(r"(\d{2})[/.](\d{2})[/.](\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?")

_OCR = None


def _ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def _read_wallclock(frame) -> float | None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = HEADER_CROP
    crop = frame[int(y0 * h):int(y1 * h) or 1, int(x0 * w):int(x1 * w)]
    crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 3))
    res = _ocr()(np.asarray(crop), use_det=True, use_cls=False, use_rec=True)
    text = " ".join(t[1] for t in (res[0] or [])) if res and res[0] else ""
    m = TS_RE.search(text)
    if not m:
        return None
    d, mo, y, H, M, S, ms = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d), int(H), int(M), int(S),
                      int((ms or "0").ljust(3, "0")) * 1000)
        return dt.timestamp()
    except ValueError:
        return None


def _duration(video: str) -> float:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return float(out.stdout.strip() or 0.0)


def wallclock_series(video: str, n: int = 40):
    """[(video_time_s, wallclock_epoch_s)] from n frames spread over the video.
    Uses ffmpeg for accurate per-timestamp seeks (cv2 POS_FRAMES seeks land
    on keyframes / return decode garbage on these long-GOP files)."""
    import subprocess
    import tempfile

    dur = _duration(video)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(n):
            t = dur * (k + 0.5) / n
            p = f"{tmp}/f.png"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t}", "-i", str(video),
                            "-frames:v", "1", p], check=False)
            frame = cv2.imread(p)
            if frame is None:
                continue
            wc = _read_wallclock(frame)
            if wc is not None:
                out.append((t, wc))
    return out


def offset(video_a: str, video_b: str, n: int = 40):
    """seconds to ADD to A's timeline to land on B's timeline
    (b_time = a_time + offset). Both videos play at ~1x, so A's video-time
    is a near-linear function of wall-clock; fit it and, for each B sample,
    diff B's video-time against A's video-time at the same wall-clock."""
    sa = wallclock_series(video_a, n)
    sb = wallclock_series(video_b, n)
    if len(sa) < 2 or len(sb) < 2:
        raise RuntimeError(f"header OCR too sparse (A={len(sa)}, B={len(sb)} readable frames)")

    wa = np.array([w for _, w in sa])
    ta = np.array([t for t, _ in sa])
    w0 = wa.min()
    wa = wa - w0                                  # keep the fit well-conditioned
    # Theil-Sen: robust against the odd frozen-clock OCR misread
    ii, jj = np.triu_indices(len(wa), k=1)
    dw = wa[jj] - wa[ii]
    ok = np.abs(dw) > 1.0
    slope = float(np.median((ta[jj] - ta[ii])[ok] / dw[ok]))
    icpt = float(np.median(ta - slope * wa))     # a_video_t ~= slope*(wallclock-w0) + icpt

    diffs, used = [], []
    for tb, wb in sb:
        if wa.min() - 10 <= wb - w0 <= wa.max() + 10:
            a_t = slope * (wb - w0) + icpt
            diffs.append(tb - a_t)
            used.append({"wallclock": round(wb, 3), "a_t": round(float(a_t), 3),
                        "b_t": round(tb, 3), "b_minus_a": round(float(tb - a_t), 3)})
    if not diffs:
        raise RuntimeError("A and B wall-clock ranges don't overlap")
    d = np.array(diffs)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) or 1e-6
    keep = np.abs(d - med) <= 4 * mad
    used = [u for u, k in zip(used, keep) if k]
    return float(np.median(d[keep])), used


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="reference video (e.g. the face cam)")
    ap.add_argument("--b", required=True, help="video to align to A (e.g. the ATM cam)")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    off, samples = offset(args.a, args.b, args.n)
    out = ATM_DIR / args.sortie / "sync.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "a": args.a, "b": args.b,
        "offset_b_minus_a_s": round(off, 3),
        "usage": "b_time = a_time + offset_b_minus_a_s",
        "n_samples": len(samples), "samples": samples[:20],
    }, indent=2))
    spread = (max(s["b_minus_a"] for s in samples) - min(s["b_minus_a"] for s in samples)
              if samples else 0)
    print(f"offset (B - A) = {off:+.3f} s   from {len(samples)} samples, spread {spread:.3f} s")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
