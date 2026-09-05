"""One comparison movie: the video plays once, and a stacked panel shows
every diarization backend's current speaker decision side by side, in
sync -- so you can see where they agree and disagree.

Feeds off the per-backend labelled transcripts that diarize_compare.py
writes to exports/audio/<sortie>/diarize_compare/<backend>.transcript.json.

    python3 code/speech/diarize_vis_multi.py \\
        --dir "exports/audio/s1 c2/diarize_compare" \\
        --video "exports/audio/s1 c2/diarize_compare/window.mp4"

Each backend row: name + a coloured chip for the speaker talking at the
current frame (SPEAKER_00 = blue, 01 = amber, 02 = green, 03 = purple,
UNK / silence = grey). The bottom strip carries the transcript subtitle
from whichever backend file is listed first (they share the same text,
only the speaker labels differ).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from tqdm import tqdm

_HAS_RAQM = features.check("raqm")   # RAQM does bidi+shaping; don't pre-reorder
if not _HAS_RAQM:
    from bidi.algorithm import get_display
else:
    def get_display(s):
        return s

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Transcript  # noqa: E402

FONT = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
SPK_RGBA = {
    "SPEAKER_00": (39, 99, 168, 255), "SPEAKER_01": (227, 122, 39, 255),
    "SPEAKER_02": (46, 160, 92, 255), "SPEAKER_03": (139, 92, 224, 255),
    "SPEAKER_04": (198, 63, 47, 255), "SPEAKER_05": (222, 122, 179, 255),
    "SIM_EN": (20, 175, 200, 255), "SIM_HE": (120, 175, 40, 255),
    "SPEAKER_UNK": (110, 110, 110, 255), None: (70, 70, 70, 255),
}


def _spk_color(spk):
    if spk in SPK_RGBA:
        return SPK_RGBA[spk]
    if spk and spk.startswith("SIM_"):
        return SPK_RGBA["SIM_EN"]
    return SPK_RGBA["SPEAKER_UNK"]
ORDER = ["mfcc", "nemo", "pyannote", "speechbrain", "wespeaker", "3dspeaker", "whisperx", "kaldi"]
HEBREW = tuple(range(0x0590, 0x0600)) + tuple(range(0xFB1D, 0xFB50))


def _is_rtl(s):
    return any(ord(c) in HEBREW for c in s)


def _speaker_at(segs, t):
    for s in segs:
        if s.start <= t <= s.end:
            return s.speaker
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="diarize_compare output dir")
    ap.add_argument("--video", required=True, help="the window .mp4 in that dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    files = {p.name.split(".")[0]: p for p in d.glob("*.transcript.json")}
    backends = [b for b in ORDER if b in files] + [b for b in files if b not in ORDER]
    if not backends:
        sys.exit(f"no <backend>.transcript.json files in {d}")
    tracks = {b: Transcript.load(files[b]) for b in backends}
    base = tracks[backends[0]]
    print(f"backends: {backends}")

    # re-encode to constant frame rate first so frame count and audio length
    # are locked together (the window.mp4 was cut with -c copy = VFR -> drift)
    from audio_vis import _cfr_window
    p = cv2.VideoCapture(args.video)
    raw_fps = p.get(cv2.CAP_PROP_FPS)
    nfr = p.get(cv2.CAP_PROP_FRAME_COUNT)
    p.release()
    if not raw_fps or raw_fps <= 0 or raw_fps > 240:
        sys.exit(f"could not read a sane fps ({raw_fps}) from {args.video}")
    src_fps = round(raw_fps, 3)
    src_dur = (nfr or 0) / src_fps
    cfr = _cfr_window(args.video, 0, src_dur, src_fps)

    cap = cv2.VideoCapture(str(cfr))
    fps = src_fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    panel_h = 34 * len(backends) + 24
    sub_h = 96
    out_h = h + panel_h + sub_h
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 19)

    out_path = Path(args.out or d / "compare_all.mp4")
    silent = out_path.with_suffix(".silent.mp4")
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, out_h))

    for i in tqdm(range(n_frames), desc="rendering", unit="f"):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        canvas = np.zeros((out_h, w, 3), np.uint8)
        canvas[:h] = frame
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, "RGBA")

        y = h + 12
        for b in backends:
            spk = _speaker_at(tracks[b].segments, t)
            col = _spk_color(spk)
            draw.text((16, y + 4), b, font=small, fill=(230, 230, 230, 255))
            draw.rectangle([150, y, 150 + 230, y + 26], fill=col)
            draw.text((160, y + 4), spk or "-", font=small, fill=(255, 255, 255, 255))
            y += 34

        cur = next((s for s in base.segments if s.start <= t <= s.end and s.text.strip()), None)
        if cur:
            logical = cur.text.strip()
            rtl = _is_rtl(logical)
            while draw.textbbox((0, 0), get_display(logical), font=font)[2] > w - 40 \
                    and len(logical) > 8:
                logical = logical[:-2] if not rtl else logical[2:]   # trim from the reading tail
            shaped = get_display(logical)
            bb = draw.textbbox((0, 0), shaped, font=font)
            tw = bb[2] - bb[0]
            x = (w - 20 - tw) if rtl else 20
            draw.rectangle([0, out_h - sub_h, w, out_h], fill=(0, 0, 0, 170))
            draw.text((x, out_h - sub_h + 30), shaped, font=font, fill=(255, 255, 255, 255),
                     direction="rtl" if (rtl and _HAS_RAQM) else "ltr")

        vw.write(cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR))

    vw.release()
    cap.release()
    print("muxing audio ...")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(cfr),
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
                    "-crf", "20", "-c:a", "aac", "-shortest", str(out_path)], check=True)
    silent.unlink(missing_ok=True)
    Path(cfr).unlink(missing_ok=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
