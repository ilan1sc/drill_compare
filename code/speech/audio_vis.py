"""Burn diarization + transcript back onto the source video: a colored
speaker banner (cadet = left, instructor = right) plus bottom subtitles.

    python3 code/speech/audio_vis.py "/home/ilan/Documents/flight school/data/s1 c2.mp4" \\
        --transcript "exports/audio/s1 c2/transcript.json" --start 0 --duration 45

Renders frame-by-frame with OpenCV + PIL (PIL for the subtitle text --
cv2.putText can't shape Hebrew), muxes the ORIGINAL audio for that window
back in with ffmpeg at the end (the silent frame-by-frame pass has no
audio of its own). Output defaults to exports/audio/<sortie>/vis_sample.mp4.

Speaker attribution: whichever segment (transcript.json, already
diarized+glossary-tagged by code/speech/pipeline.py) covers the current
frame's timestamp. --cadet-speaker picks which SPEAKER_xx label is drawn
on the left; defaults to the busier of the two speakers (SPEAKER_UNK,
diarize.py's low-confidence bucket, is excluded from that count). The
other speaker goes on the right, labeled INSTRUCTOR.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from tqdm import tqdm

# Pillow here is built with libraqm, so ImageFont.truetype() uses the RAQM
# layout engine, which does full Unicode bidi + shaping itself. Feeding it
# python-bidi's get_display() output double-applies bidi and scrambles the
# text -- so we pass LOGICAL text straight through and let RAQM lay it out,
# only choosing left/right alignment ourselves. Fall back to python-bidi
# reordering only if RAQM isn't available.
_HAS_RAQM = features.check("raqm")
if not _HAS_RAQM:
    from bidi.algorithm import get_display
else:
    def get_display(s):  # no-op: RAQM handles bidi
        return s

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Transcript  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"
FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"  # has Hebrew glyphs

CADET_RGBA = (39, 99, 168, 255)    # blue -- same hue as --cadet in transcript_review.html
INSTR_RGBA = (227, 122, 39, 255)   # amber -- same hue as --instr in transcript_review.html
BAR_W = 14
SUB_PAD = 18


HEBREW = tuple(range(0x0590, 0x0600)) + tuple(range(0xFB1D, 0xFB50))


def _is_rtl(s: str) -> bool:
    return any(ord(c) in HEBREW for c in s)


def _wrap_logical(draw, text, font, max_w):
    """Wrap on the logical string; RAQM handles bidi at draw time."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textbbox((0, 0), get_display(trial), font=font)[2] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _cfr_window(video, start, duration, fps) -> Path:
    """One constant-frame-rate clip carrying BOTH the exact frames we'll
    draw on and the exact audio for the same span -- so nothing can drift."""
    out = Path(video).with_suffix("")
    out = out.parent / f".{out.name}_cfr_{int(start)}_{int(duration)}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(duration),
        "-i", str(video), "-r", f"{fps}", "-vsync", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", str(out),
    ], check=True)
    return out


def render(video: str, transcript_json: str, start: float, duration: float,
          out_path: str, cadet_speaker: str | None = None) -> Path:
    t = Transcript.load(transcript_json)
    counts: dict[str, int] = {}
    for w in t.words():
        if w.speaker and w.speaker != "SPEAKER_UNK":
            counts[w.speaker] = counts.get(w.speaker, 0) + 1
    ranked = sorted(counts, key=counts.get, reverse=True)
    cadet = cadet_speaker or (ranked[0] if ranked else None)
    instr = next((s for s in ranked if s != cadet), None)
    print(f"cadet={cadet}  instructor={instr}  (word counts: {counts})")

    segs = sorted(t.segments, key=lambda s: s.start)

    def active_segment(tt):
        for s in segs:
            if s.start <= tt <= s.end:
                return s
        return None

    probe = cv2.VideoCapture(video)
    raw_fps = probe.get(cv2.CAP_PROP_FPS)
    probe.release()
    if not raw_fps or raw_fps <= 0 or raw_fps > 240:
        raise RuntimeError(f"could not read a sane fps ({raw_fps}) from {video}")
    fps = round(raw_fps, 3)

    cfr = _cfr_window(video, start, duration, fps)   # frames + audio, locked together
    cap = cv2.VideoCapture(str(cfr))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    font = ImageFont.truetype(FONT_PATH, max(22, h // 32))
    label_font = ImageFont.truetype(FONT_PATH, max(20, h // 36))

    tmp_silent = Path(out_path).with_suffix(".silent.mp4")
    vw = cv2.VideoWriter(str(tmp_silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i in tqdm(range(n_frames), desc="rendering frames", unit="f"):
        ok, frame = cap.read()
        if not ok:
            break
        tt = i / fps                       # cfr clip starts at 0; transcript is already windowed
        seg = active_segment(tt)
        speaker = seg.speaker if seg else None

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, "RGBA")

        if speaker == cadet:
            draw.rectangle([0, 0, BAR_W, h], fill=CADET_RGBA)
            draw.rectangle([0, 0, 220, 46], fill=CADET_RGBA[:3] + (200,))
            draw.text((16, 10), "CADET", font=label_font, fill=(255, 255, 255, 255))
        elif speaker == instr:
            draw.rectangle([w - BAR_W, 0, w, h], fill=INSTR_RGBA)
            draw.rectangle([w - 260, 0, w, 46], fill=INSTR_RGBA[:3] + (200,))
            bbox = draw.textbbox((0, 0), "INSTRUCTOR", font=label_font)
            draw.text((w - 16 - (bbox[2] - bbox[0]), 10), "INSTRUCTOR", font=label_font,
                     fill=(255, 255, 255, 255))

        if seg and seg.text.strip():
            logical = seg.text.strip()
            rtl = _is_rtl(logical)
            lines = _wrap_logical(draw, logical, font, w - 2 * SUB_PAD)[-3:]
            line_h = font.size + 10
            box_h = line_h * len(lines) + 2 * SUB_PAD
            draw.rectangle([0, h - box_h, w, h], fill=(0, 0, 0, 150))
            y = h - box_h + SUB_PAD
            for ln in lines:
                ln = get_display(ln)     # no-op with RAQM; bidi-reorders only on fallback
                bb = draw.textbbox((0, 0), ln, font=font)
                tw = bb[2] - bb[0]
                x = (w - SUB_PAD - tw) if rtl else SUB_PAD  # RTL: right-align; LTR: left-align
                draw.text((x, y), ln, font=font, fill=(255, 255, 255, 255),
                         direction="rtl" if (rtl and _HAS_RAQM) else "ltr")
                y += line_h

        vw.write(cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR))

    vw.release()
    cap.release()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("muxing audio (ffmpeg) ...")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(tmp_silent), "-i", str(cfr),      # audio straight from the CFR window
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-shortest", str(out_path),
    ], check=True)
    tmp_silent.unlink(missing_ok=True)
    cfr.unlink(missing_ok=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sortie = Path(args.transcript).parent.name
    out = args.out or str(AUDIO_DIR / sortie / "vis_sample.mp4")
    result = render(args.video, args.transcript, args.start, args.duration, out,
                    cadet_speaker=args.cadet_speaker)
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
