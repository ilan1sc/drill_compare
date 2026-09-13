"""Read the Garmin PFD readouts from a video and log them per frame.

Give it a movie (and optionally a begin/end time). For every frame in
that range it reads five numbers from fixed screen regions and appends a
row to a CSV:

    frame, time_s, altitude, ra, heading, pi

Output goes to exports/pfd/<video-stem>_pfd.csv.

The regions of interest are fractions of the frame size in ROIS below,
tuned to a 1920x1080 PFD layout. If your video differs, run with --debug
to dump each crop from the first frame to exports/pfd/debug/ and adjust,
or pass --rois myrois.json (same shape; values >1 are absolute pixels).

Speed: frames are streamed straight from ffmpeg as 8-bit gray (no
intermediate files), OCR is recognition-only on five tiny crops, and a
crop whose pixels barely changed from the previous frame (mean abs diff
below --still) reuses the last value instead of running OCR again -- on a
60 fps PFD, where a readout holds for dozens of frames, that skips the
large majority of calls without ever guessing between real changes.
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

# box: (x, y, w, h) as fractions of the frame. lo/hi: plausible value range.
# jump: with --smooth, a reading this far from the local trend is discarded
# as an OCR error (units per field, e.g. kt / ft / deg).
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
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
            "-of", "json", str(path),
        ],
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
    """Discard readings far from the local trend, then median-smooth.

    Handles both lone spikes and multi-frame misreads (e.g. the airspeed
    ones-digit drum blurring mid-roll): a wide median gives the trend, any
    sample more than `jump` from it is dropped, then a short median tidies
    the rest.
    """
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

    ap = argparse.ArgumentParser(description="Log Garmin PFD readouts per frame.")
    ap.add_argument("video")
    ap.add_argument("begin", nargs="?", help="begin time (seconds or HH:MM:SS); default start")
    ap.add_argument("end", nargs="?", help="end time (seconds or HH:MM:SS); default end")
    ap.add_argument("--scale", type=int, default=6, help="ROI upscale factor for OCR (default 6)")
    ap.add_argument("--still", type=float, default=1.5,
                    help="mean abs pixel diff below which a crop counts as unchanged (default 1.5)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="optional median-filter window (frames) applied to each column; 0 = off, log raw per-frame values")
    ap.add_argument("--rois", help="JSON file overriding regions of interest")
    ap.add_argument("--out", help="output CSV path")
    ap.add_argument("--debug", action="store_true", help="dump first-frame ROI crops to exports/pfd/debug/")
    args = ap.parse_args()

    src = Path(args.video).expanduser()
    if not src.is_file():
        sys.exit(f"Not a file: {src}")

    W, H, fps, dur = probe(src)
    if not fps:
        sys.exit("could not read frame rate")
    start = parse_time(args.begin) if args.begin else 0.0
    end = parse_time(args.end) if args.end else dur
    if end <= start:
        sys.exit("end time must be after begin time")
    start_frame = round(start * fps)
    rois = load_rois(args.rois)
    fields = list(rois)  # a --rois override can add fields beyond the FIELDS default
    boxes = {n: px_box(rois[n]["box"], W, H) for n in fields}

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out) if args.out else EXPORT_DIR / f"{src.stem}_pfd.csv"
    if args.debug:
        (EXPORT_DIR / "debug").mkdir(exist_ok=True)

    engine = RapidOCR()
    frame_bytes = W * H
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error",
            "-ss", f"{start}", "-to", f"{end}",
            "-i", str(src),
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        stdout=subprocess.PIPE,
    )

    last_sig = {n: None for n in fields}
    last_val = {n: None for n in fields}
    series = {n: [] for n in fields}
    n_ocr = 0
    est_total = int(round((end - start) * fps))

    i = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W)
        for n in fields:
            x, y, w, h = boxes[n]
            crop = frame[y:y + h, x:x + w]
            if args.debug and i == 0:
                Image.fromarray(crop).save(EXPORT_DIR / "debug" / f"{n}.png")
            sig = crop.astype(np.int16)
            unchanged = last_sig[n] is not None and \
                np.abs(sig - last_sig[n]).mean() < args.still
            last_sig[n] = sig
            if not unchanged:
                last_val[n] = read_int(
                    engine, crop, rois[n]["lo"], rois[n]["hi"], args.scale
                )
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
            fnum = start_frame + k
            fh.write(f"{fnum},{fnum / fps:.3f}," + ",".join(
                "" if series[n][k] is None else str(series[n][k]) for n in fields
            ) + "\n")

    print(f"\n{i} frames -> {out_csv}")
    print(f"{n_ocr} OCR calls ({n_ocr / max(i * len(fields), 1):.1%} of crops)")


if __name__ == "__main__":
    main()
