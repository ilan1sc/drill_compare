"""Run several diarization backends on the SAME short window and render a
speaker-labelled movie for each, so they can be compared side by side.

    python3 code/speech/diarize_compare.py "/home/ilan/Documents/flight school/data/s1 c2.mp4" \\
        --transcript "exports/audio/s1 c2/transcript.json" --start 0 --duration 300 \\
        --backends mfcc,nemo,speechbrain

For each backend it:
  1. cuts a <duration>s window of audio + video (rebased to t=0),
  2. copies the transcript window, times rebased,
  3. runs that backend's diarize(transcript, wav) -> speaker labels,
  4. saves the labelled transcript and renders a movie with audio_vis.py
     (cadet banner left / instructor right + subtitles).

Outputs under exports/audio/<sortie>/diarize_compare/:
    <backend>.transcript.json   <backend>.mp4
    _report.txt   (per-backend speaker word-counts + any that were skipped)

Backend keys: mfcc (the original diarize.py), nemo, pyannote, speechbrain,
wespeaker, 3dspeaker, whisperx, kaldi. Ones needing a HF token / extra
install / a Kaldi tree are skipped with a note rather than erroring.
"""

import argparse
import importlib
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Transcript  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"

BACKENDS = {
    "mfcc": ("diarize", "diarize"),          # original diarize.py (module, function)
    "nemo": ("diarize_nemo", "diarize"),
    "pyannote": ("diarize_pyannote", "diarize"),
    "speechbrain": ("diarize_speechbrain", "diarize"),
    "wespeaker": ("diarize_wespeaker", "diarize"),
    "3dspeaker": ("diarize_3dspeaker", "diarize"),
    "whisperx": ("diarize_whisperx", "diarize"),
    "kaldi": ("diarize_kaldi", "diarize"),
}


def _window_transcript(t: Transcript, start: float, end: float) -> Transcript:
    segs = []
    for s in t.segments:
        if s.end < start or s.start > end:
            continue
        ws = [w for w in s.words if start <= (w.start + w.end) / 2 <= end]
        s2 = type(s)(id=len(segs), start=max(0.0, s.start - start), end=max(0.0, s.end - start),
                     text=s.text, lang=s.lang, speaker=None,
                     words=[type(w)(text=w.text, start=w.start - start, end=w.end - start,
                                    prob=w.prob) for w in ws])
        segs.append(s2)
    return Transcript(sortie=t.sortie, source_video=t.source_video,
                      duration=end - start, segments=segs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--backends", default="mfcc,nemo,speechbrain,wespeaker,3dspeaker")
    ap.add_argument("--speakers", default="auto",
                    help="max speaker count for backends that accept one "
                         "(int, or 'auto' to let each backend decide). Up to ~5 here: "
                         "cadet, instructor, sim instructor, HE sim callouts, EN sim callouts.")
    ap.add_argument("--langid", action="store_true",
                    help="after each backend, run diarize_langid.refine() to split out "
                         "SIM_EN callout segments (unreliable on short noisy clips -- see that module)")
    ap.add_argument("--no-video", action="store_true", help="skip movie rendering, labels only")
    args = ap.parse_args()
    n_spk = None if args.speakers == "auto" else int(args.speakers)

    base = Transcript.load(args.transcript)
    sortie = Path(args.transcript).parent.name
    out_dir = AUDIO_DIR / sortie / "diarize_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = args.start, args.start + args.duration

    win_wav = out_dir / "window.wav"
    win_mp4 = out_dir / "window.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(args.duration),
                    "-i", args.video, "-ac", "1", "-ar", "16000", str(win_wav)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(args.duration),
                    "-i", args.video, "-c", "copy", str(win_mp4)], check=True)

    report = []
    for key in [b.strip() for b in args.backends.split(",") if b.strip()]:
        mod_name, fn_name = BACKENDS[key]
        print(f"\n=== {key} ({mod_name}) ===")
        wt = _window_transcript(base, start, end)
        try:
            fn = getattr(importlib.import_module(mod_name), fn_name)
            try:
                fn(wt, win_wav, n_speakers=n_spk)
            except TypeError:
                fn(wt, win_wav)               # backend doesn't take a count (nemo, wespeaker)
            if args.langid:
                importlib.import_module("diarize_langid").refine(wt, win_wav)
            counts: dict[str, int] = {}
            for w in wt.words():
                counts[w.speaker] = counts.get(w.speaker, 0) + 1
            tj = out_dir / f"{key}.transcript.json"
            wt.save(tj)
            report.append(f"{key:12} OK   {counts}")
            print(f"  {counts} -> {tj}")
            if not args.no_video:
                subprocess.run([sys.executable, str(Path(__file__).parent / "audio_vis.py"),
                                str(win_mp4), "--transcript", str(tj), "--start", "0",
                                "--duration", str(args.duration),
                                "--out", str(out_dir / f"{key}.mp4")], check=True)
                print(f"  movie -> {out_dir / f'{key}.mp4'}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}".splitlines()[0][:200]
            report.append(f"{key:12} SKIP {msg}")
            print(f"  SKIPPED: {msg}")
            traceback.print_exc(limit=1)

    (out_dir / "_report.txt").write_text(
        f"diarize_compare  {sortie}  window {start:.0f}-{end:.0f}s\n\n" + "\n".join(report) + "\n")
    print(f"\nreport -> {out_dir / '_report.txt'}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
