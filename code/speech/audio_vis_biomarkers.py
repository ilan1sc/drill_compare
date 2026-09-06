"""Like audio_vis.py (speaker banner + Hebrew subtitles) but with a
BIOMARKER panel down the right edge: every b01..b15 result's `evidence`
rows are indexed by time, and while the current utterance is on screen
the biomarkers that fired on it light up (with the matched tag).

    python3 code/speech/audio_vis_biomarkers.py "/home/ilan/Documents/flight school/data/s1 c2.mp4" \\
        --dir "exports/audio_best/s1 c2" --start 0 --duration 180

Reads <dir>/transcript.json and <dir>/biomarkers/b*.json. Output ->
<dir>/biomarker_vis_<start>_<dur>.mp4 (or --out).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Transcript  # noqa: E402
from audio_vis import _HAS_RAQM, FONT_PATH, HEBREW, _is_rtl, get_display  # noqa: E402
from av_sync import accurate_cfr, render_and_sync  # noqa: E402

CADET_RGBA = (39, 99, 168, 255)
INSTR_RGBA = (227, 122, 39, 255)
HIT_RGBA = (46, 160, 92, 255)
PANEL_W = 430
BAR_W = 14

LABELS = {
    "b01_observation_statements": "observation", "b02_intention_statements": "intention",
    "b03_action_statements": "action", "b04_verification_statements": "verification",
    "b05_oiav_loop": "OIAV loop", "b06_error_recognition_correction": "error correction",
    "b07_prediction_statements": "prediction", "b08_prioritization_statements": "prioritization",
    "b09_information_sharing_crm": "info sharing / CRM", "b10_communication_relevance": "comm relevance",
    "b11_verbalization_by_phase": "verbalization/phase", "b12_plan_clarity": "plan clarity",
    "b13_disfluencies_restarts": "disfluency / restart", "b14_pause_hesitation": "pause / hesitation",
    "b15_acoustic_candidates": "acoustic candidate", "b16_learning_progression": "learning progression",
}


def _load_events(bdir: Path):
    """{module_name: [(t, tag), ...]} from every biomarker json's evidence."""
    ev = {}
    for p in sorted(bdir.glob("b*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        rows = d.get("evidence") or []
        ev[p.stem] = sorted((float(r.get("t", 0.0)), str(r.get("tag", "") or "")) for r in rows)
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--dir", required=True, help="exports/audio_best/<sortie>")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-auto-sync", action="store_true",
                    help="skip the automatic A/V sync check + correction")
    ap.add_argument("--verify-n", type=int, default=12,
                    help="random points to sample for the sync check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.dir)
    t = Transcript.load(d / "transcript.json")
    events = _load_events(d / "biomarkers")
    mods = [m for m in LABELS if m in events]

    counts: dict[str, int] = {}
    for w in t.words():
        if w.speaker and w.speaker != "SPEAKER_UNK":
            counts[w.speaker] = counts.get(w.speaker, 0) + 1
    ranked = sorted(counts, key=counts.get, reverse=True)
    cadet = args.cadet_speaker or (ranked[0] if ranked else None)
    instr = next((s for s in ranked if s != cadet), None)

    start, dur = args.start, args.duration
    segs = sorted((s for s in t.segments if s.end >= start and s.start <= start + dur),
                  key=lambda s: s.start)

    probe = cv2.VideoCapture(args.video)
    raw_fps = probe.get(cv2.CAP_PROP_FPS)
    probe.release()
    if not raw_fps or raw_fps <= 0 or raw_fps > 240:
        sys.exit(f"bad fps {raw_fps}")
    fps = round(raw_fps, 3)
    cfr = accurate_cfr(args.video, start, dur, fps)   # accurate seek -> frame i is at start + i/fps
    cap = cv2.VideoCapture(str(cfr))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(round(dur * fps))                  # deterministic, not CAP_PROP_FRAME_COUNT
    outw = w + PANEL_W

    font = ImageFont.truetype(FONT_PATH, max(22, h // 32))
    lab_font = ImageFont.truetype(FONT_PATH, max(20, h // 36))
    row_font = ImageFont.truetype(FONT_PATH, 20)
    small = ImageFont.truetype(FONT_PATH, 16)

    out_path = Path(args.out or d / f"biomarker_vis_{int(start)}_{int(dur)}.mp4")

    fired_total = {m: 0 for m in mods}
    seen_ev = {m: set() for m in mods}

    def frames():
      last = np.zeros((h, outw, 3), np.uint8)
      for i in tqdm(range(n_frames), desc="biomarker-vis", unit="f"):
        ok, frame = cap.read()
        if not ok:
            yield last                                # pad so video length == audio length
            continue
        tt = start + i / fps
        seg = next((s for s in segs if s.start <= tt <= s.end), None)
        speaker = seg.speaker if seg else None

        canvas = np.zeros((h, outw, 3), np.uint8)
        canvas[:, :w] = frame
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, "RGBA")

        if speaker == cadet:
            draw.rectangle([0, 0, BAR_W, h], fill=CADET_RGBA)
            draw.rectangle([0, 0, 220, 46], fill=CADET_RGBA[:3] + (200,))
            draw.text((16, 10), "CADET", font=lab_font, fill=(255, 255, 255, 255))
        elif speaker == instr:
            draw.rectangle([w - BAR_W, 0, w, h], fill=INSTR_RGBA)
            draw.rectangle([w - 260, 0, w, 46], fill=INSTR_RGBA[:3] + (200,))
            bb = draw.textbbox((0, 0), "INSTRUCTOR", font=lab_font)
            draw.text((w - 16 - (bb[2] - bb[0]), 10), "INSTRUCTOR", font=lab_font,
                      fill=(255, 255, 255, 255))

        if seg and seg.text.strip():
            logical = seg.text.strip()
            rtl = _is_rtl(logical)
            line = get_display(logical)
            while draw.textbbox((0, 0), line, font=font)[2] > w - 40 and len(logical) > 8:
                logical = logical[2:] if rtl else logical[:-2]
                line = get_display(logical)
            bb = draw.textbbox((0, 0), line, font=font)
            x = (w - 20 - (bb[2] - bb[0])) if rtl else 20
            draw.rectangle([0, h - 88, w, h], fill=(0, 0, 0, 160))
            draw.text((x, h - 62), line, font=font, fill=(255, 255, 255, 255),
                      direction="rtl" if (rtl and _HAS_RAQM) else "ltr")

        draw.rectangle([w, 0, outw, h], fill=(18, 18, 22, 255))
        draw.text((w + 14, 12), f"BIOMARKERS   t={tt:6.1f}s", font=small,
                  fill=(180, 180, 190, 255))
        y = 44
        win = (seg.start - 0.4, seg.end + 0.4) if seg else (tt - 1.5, tt + 1.5)
        for m in mods:
            hot_tag = None
            for et, tag in events[m]:
                if win[0] <= et <= win[1]:
                    hot_tag = tag or "hit"
                    if et not in seen_ev[m]:
                        seen_ev[m].add(et)
                        fired_total[m] += 1
            row = LABELS[m]
            if hot_tag:
                draw.rectangle([w + 8, y - 2, outw - 8, y + 24], fill=HIT_RGBA[:3] + (235,))
                draw.text((w + 14, y), row, font=row_font, fill=(255, 255, 255, 255))
                tg = hot_tag[:22]
                bb = draw.textbbox((0, 0), tg, font=small)
                draw.text((outw - 14 - (bb[2] - bb[0]), y + 3), tg, font=small,
                          fill=(235, 255, 235, 255))
            else:
                draw.text((w + 14, y), row, font=row_font, fill=(120, 120, 130, 255))
            draw.text((outw - 40, y + 3), str(fired_total[m]), font=small,
                      fill=(150, 150, 160, 255))
            y += 30

        last = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        yield last

    rep = render_and_sync(frames(), outw, h, fps, args.video, start, dur, out_path,
                          auto=not args.no_auto_sync, verify_n=args.verify_n, seed=args.seed)
    cap.release()
    Path(cfr).unlink(missing_ok=True)
    (out_path.parent / (out_path.stem + ".sync.json")).write_text(
        json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
