"""Pull the audio track out of a sortie video as 16kHz mono WAV (what
whisper wants). Run standalone or import extract().

    python3 code/speech/extract_audio.py "/path/s1 c2.mp4"
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"


def extract(video: str | Path, out_dir: Path = AUDIO_DIR) -> Path:
    video = Path(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{video.stem}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(wav)],
        check=True,
    )
    return wav


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--out-dir", default=str(AUDIO_DIR))
    args = ap.parse_args()
    wav = extract(args.video, Path(args.out_dir))
    print(f"Wrote {wav}")


if __name__ == "__main__":
    main()
