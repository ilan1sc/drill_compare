"""Same I/O as pfd.py (video in, frame/time_s/<fields> csv out) -- with
extra knobs to trade OCR volume for speed. All the actual per-crop OCR
logic (read_int, still-frame skip, clean_series) is identical to pfd.py;
the difference is how many frames/crops it bothers to look at.

Shortcuts, independently selectable:

  --time-hz H     decode at H fps via ffmpeg's own -r resampling instead
                   of the video's native fps -- fewer frames ever reach
                   the pipe at all (cheaper decode, not just cheaper OCR)
  --frame-stride K   decode every frame (native fps) but only run the
                   still-check/OCR on every Kth one; skipped frames just
                   repeat the last value
  --scale S        ROI upscale factor before OCR (pfd.py default 6) --
                   smaller crops for the OCR model to chew on
  --still X        mean-abs-diff threshold for "unchanged" (pfd.py
                   default 1.5) -- raise it to let the existing skip
                   logic itself skip more aggressively

--time-hz and --frame-stride are mutually exclusive (both are "look at
fewer samples", just decode-side vs. OCR-side); --scale/--still combine
with either.

    python3 code/pfd_sync/pfd2.py <video> --time-hz 5 --out out.csv
    python3 code/pfd_sync/pfd2.py <video> --frame-stride 4
    python3 code/pfd_sync/pfd2.py <video> --scale 3 --still 3.0
    python3 code/pfd_sync/pfd2.py <video> --preset fast   # --scale 2 --still 6.0

Of everything benchmarked (see code/pfd_sync/README.md), --scale/--still
beat --time-hz/--frame-stride at the same speedup: shrinking the OCR
crop or loosening the "did this change" threshold barely moves the
downstream sync accuracy, while skipping frames outright (decode- or
OCR-side) measurably does. --preset fast bundles the best of the
scale/still combinations found.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

EXPORT_DIR = Path(__file__).resolve().parent.parent.parent / "exports" / "pfd"

ROIS = {
    "altitude": {"box": (0.62292, 0.29259, 0.05000, 0.04815), "lo": -1000, "hi": 25000, "jump": 150},
    "ra":       {"box": (0.57917, 0.55370, 0.03854, 0.03704), "lo": 0,     "hi": 2500,  "jump": 80},
    "heading":  {"box": (0.44271, 0.42037, 0.05000, 0.04444), "lo": 0,     "hi": 360,   "jump": 20},
    "pi":       {"box": (0.25417, 0.41481, 0.03229, 0.04444), "lo": 0,     "hi": 130,   "jump": 15},
}
FIELDS = list(ROIS)


def parse_time(s: str) -> float:
    s = s.strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    return float(s)


def probe(path: Path) -> tuple[int, int, float, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    st = data["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    dur = float(data.get("format", {}).get("duration", 0.0))
    return int(st["width"]), int(st["height"]), fps, dur


def load_rois(path: str | None) -> dict:
    if not path:
        return ROIS
    user = json.loads(Path(path).read_text())
    merged = {k: dict(v) for k, v in ROIS.items()}
    for k, v in user.items():
        merged.setdefault(k, {"lo": -1e9, "hi": 1e9})
        merged[k].update(v)
    return merged


def px_box(box, W, H) -> tuple[int, int, int, int]:
    x, y, w, h = box
    if max(box) <= 1.0:
        x, y, w, h = x * W, y * H, w * W, h * H
    return int(x), int(y), int(w), int(h)


def read_int(engine: RapidOCR, crop: np.ndarray, lo, hi, scale: int):
    h, w = crop.shape
    big = Image.fromarray(crop).resize((w * scale, h * scale), Image.LANCZOS)
    res = engine(np.asarray(big), use_det=False, use_cls=False, use_rec=True)
    text = res[0][0][0] if res and res[0] else ""
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    val = int(digits)
    return val if lo <= val <= hi else None


def _running_median(series: list, half: int) -> list:
    out = []
    for i in range(len(series)):
        seg = sorted(v for v in series[max(0, i - half):i + half + 1] if v is not None)
        out.append(seg[len(seg) // 2] if seg else None)
    return out


def clean_series(series: list, window: int, jump: float) -> list:
    trend = _running_median(series, max(window, 30))
    kept = [
        v if (v is not None and trend[i] is not None and abs(v - trend[i]) <= jump)
        else None
        for i, v in enumerate(series)
    ]
    return _running_median(kept, window // 2)


def main() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg and ffprobe must be installed and on PATH.")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("begin", nargs="?")
    ap.add_argument("end", nargs="?")
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--still", type=float, default=1.5)
    ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--rois")
    ap.add_argument("--out")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--time-hz", type=float, default=None,
                    help="decode at this fps via ffmpeg -r instead of native fps")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="only still-check/OCR every Kth native-fps frame (default 1 = every frame)")
    ap.add_argument("--preset", choices=["fast"], default=None,
                    help="fast = --scale 2 --still 6.0 (benchmarked "
                         "2026-09-13: ~1.7-3.2x faster, heading/airspeed/altitude "
                         "sync scores within noise of the unshortened baseline)")
    args = ap.parse_args()
    if args.preset == "fast":
        args.scale, args.still = 2, 6.0
    if args.time_hz and args.frame_stride > 1:
        sys.exit("--time-hz and --frame-stride are alternative shortcuts, pick one")

    src = Path(args.video).expanduser()
    if not src.is_file():
        sys.exit(f"Not a file: {src}")

    W, H, native_fps, dur = probe(src)
    if not native_fps:
        sys.exit("could not read frame rate")
    start = parse_time(args.begin) if args.begin else 0.0
    end = parse_time(args.end) if args.end else dur
    if end <= start:
        sys.exit("end time must be after begin time")

    decode_fps = args.time_hz if args.time_hz else native_fps
    rois = load_rois(args.rois)
    fields = list(rois)
    boxes = {n: px_box(rois[n]["box"], W, H) for n in fields}

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out) if args.out else EXPORT_DIR / f"{src.stem}_pfd.csv"
    if args.debug:
        (EXPORT_DIR / "debug").mkdir(exist_ok=True)

    engine = RapidOCR()
    frame_bytes = W * H
    ff_cmd = ["ffmpeg", "-v", "error", "-ss", f"{start}", "-to", f"{end}", "-i", str(src)]
    if args.time_hz:
        ff_cmd += ["-r", f"{args.time_hz}"]
    ff_cmd += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE)

    last_sig = {n: None for n in fields}
    last_val = {n: None for n in fields}
    series = {n: [] for n in fields}
    times = []
    n_ocr = 0
    est_total = int(round((end - start) * decode_fps))

    i = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        t_now = start + i / decode_fps
        times.append(t_now)
        do_ocr_pass = (i % args.frame_stride == 0)
        frame = np.frombuffer(buf, np.uint8).reshape(H, W)
        for n in fields:
            x, y, w, h = boxes[n]
            crop = frame[y:y + h, x:x + w]
            if args.debug and i == 0:
                Image.fromarray(crop).save(EXPORT_DIR / "debug" / f"{n}.png")
            if not do_ocr_pass:
                series[n].append(last_val[n])
                continue
            sig = crop.astype(np.int16)
            unchanged = last_sig[n] is not None and \
                np.abs(sig - last_sig[n]).mean() < args.still
            last_sig[n] = sig
            if not unchanged:
                last_val[n] = read_int(engine, crop, rois[n]["lo"], rois[n]["hi"], args.scale)
                n_ocr += 1
            series[n].append(last_val[n])
        i += 1
        if i % 500 == 0:
            print(f"  {i}/{est_total} frames  ({n_ocr} OCR calls)")

    proc.wait()

    if args.smooth > 1:
        series = {n: clean_series(series[n], args.smooth, rois[n].get("jump", 1e9))
                  for n in fields}

    with open(out_csv, "w") as fh:
        fh.write("frame,time_s," + ",".join(fields) + "\n")
        for k in range(i):
            fnum = round(times[k] * native_fps)
            fh.write(f"{fnum},{times[k]:.3f}," + ",".join(
                "" if series[n][k] is None else str(series[n][k]) for n in fields
            ) + "\n")

    print(f"\n{i} frames -> {out_csv}")
    print(f"{n_ocr} OCR calls ({n_ocr / max(i * len(fields), 1):.1%} of crops)")


if __name__ == "__main__":
    main()
