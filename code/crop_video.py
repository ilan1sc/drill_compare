"""Crop a rectangle out of a video and optionally trim it in time.

Run it, point it at a movie, and it prints the resolution and duration.
Then give it the horizontal pixel range (left,right), the vertical pixel
range (top,bottom), an optional start/end time, and an optional output
size. The result is written to exports/video/.

\n chunk	Answers this prompt
/home/.../s1 c2.mp4\n	Path to the movie file
0,200\n	Horizontal pixel range (left,right)
0,480\n	Vertical pixel range (top,bottom)
\n	Start time → blank = beginning
\n	End time → blank = end of movie
\n	Output size WxH → blank = keep cropped size

ex. 
cd "/home/ilan/Documents/flight school/drill_compare" && printf '/home/ilan/Documents/flight school/data/s1 c2.mp4\n0,200\n0,480\n\n\n\n' | python3 code/crop_video.py
cd "/home/ilan/Documents/flight school/drill_compare" && printf '/home/ilan/Documents/flight school/data/s1 c2.mp4\n0,200\n0,480\n0\n10\n\n' | python3 code/crop_video.py

"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

EXPORT_DIR = Path(__file__).resolve().parent.parent / "exports" / "video"


def probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    duration = float(data.get("format", {}).get("duration", 0.0))
    return int(stream["width"]), int(stream["height"]), duration


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        sys.exit("\nNo input given.")


def ask_range(prompt: str, limit: int) -> tuple[int, int]:
    raw = ask(prompt)
    parts = [p.strip() for p in raw.replace("-", ",").split(",") if p.strip()]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        sys.exit("Enter two whole numbers, e.g. 0,200")
    lo, hi = int(parts[0]), int(parts[1])
    if not (0 <= lo < hi <= limit):
        sys.exit(f"Range must satisfy 0 <= low < high <= {limit}.")
    return lo, hi


def ask_time(prompt: str) -> str | None:
    raw = ask(prompt)
    return raw or None


def main() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg and ffprobe must be installed and on PATH.")

    raw = sys.argv[1] if len(sys.argv) > 1 else ask("Path to the movie file: ")
    src = Path(raw.strip().strip('"').strip("'")).expanduser()
    if not src.is_file():
        sys.exit(f"Not a file: {src}")

    width, height, duration = probe(src)
    print(f"Resolution: {width} x {height}")
    print(f"Duration:   {duration:.2f} s")

    x0, x1 = ask_range(f"Horizontal pixel range left,right (0-{width}): ", width)
    y0, y1 = ask_range(f"Vertical pixel range top,bottom (0-{height}): ", height)
    start = ask_time("Start time (blank = beginning, e.g. 5 or 00:00:05): ")
    end = ask_time("End time (blank = end): ")
    out_size = ask("Output size WxH (blank = keep cropped size): ")

    crop_w, crop_h = x1 - x0, y1 - y0
    vf = f"crop={crop_w}:{crop_h}:{x0}:{y0}"
    if out_size:
        parts = out_size.lower().split("x")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            sys.exit("Output size must look like 320x240.")
        vf += f",scale={int(parts[0])}:{int(parts[1])}"

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    dst = EXPORT_DIR / f"{src.stem}_crop{crop_w}x{crop_h}{src.suffix}"

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if start:
        cmd += ["-ss", start]
    if end:
        cmd += ["-to", end]
    cmd += ["-vf", vf, "-c:a", "copy", str(dst)]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nExported to {dst}")


if __name__ == "__main__":
    main()
