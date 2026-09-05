"""Run several ASR backends on the SAME short window, render an audio_vis
movie for each, and write a metrics report so they can be compared.

    python3 code/speech/transcribe_compare.py "/home/ilan/Documents/flight school/data/s1 c2.mp4" \\
        --start 0 --duration 180 \\
        --backends whisper_openai,ivrit_v3,ivrit_yi,ivrit_turbo,mms,seamless_large

For each backend:
  1. transcribe the window (timed),
  2. diarize it (mfcc backend, fast -- just so the movie has speaker
     banners; text is the point here, not the speakers),
  3. save transcript.<backend>.json + render <backend>.mp4 (audio_vis),
  4. collect transcribe_common.metrics().

Outputs under exports/audio/<sortie>/asr_compare/:
    <backend>.transcript.json   <backend>.mp4
    report.md   report.json     (words / coverage / confidence /
    hallucination-repetition / hebrew-vs-latin / wall time per backend)

Backend keys: whisper_openai, ivrit_v3, ivrit_yi, ivrit_turbo, mms,
seamless_large, seamless_medium.
"""

import argparse
import importlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Transcript  # noqa: E402
from transcribe_common import metrics, timed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"

BACKENDS = {
    "whisper_openai": ("transcribe_whisper_openai", {}),
    "ivrit_v3": ("transcribe_ivrit_v3", {}),
    "ivrit_yi": ("transcribe_ivrit_yi", {}),
    "ivrit_turbo": ("transcribe_ivrit_turbo", {}),
    "mms": ("transcribe_mms", {}),
    "seamless_large": ("transcribe_seamless", {"variant": "large"}),
    "seamless_medium": ("transcribe_seamless", {"variant": "medium"}),
}
METRIC_COLS = ["words", "segments", "coverage", "words_per_min", "mean_word_prob",
               "low_conf_word_pct", "hebrew_word_pct", "latin_word_pct",
               "empty_junk_segments", "trigram_repetition", "wall_seconds",
               "realtime_factor"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--language", default="he")
    ap.add_argument("--backends",
                    default="whisper_openai,ivrit_v3,ivrit_yi,ivrit_turbo,mms,seamless_large")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="skip transcription/rendering; rebuild report.md/json from the "
                         "<backend>.transcript.json files already in asr_compare/")
    args = ap.parse_args()

    sortie = Path(args.video).stem
    out_dir = AUDIO_DIR / sortie / "asr_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        rows = {}
        for tj in sorted(out_dir.glob("*.transcript.json")):
            key = tj.name[: -len(".transcript.json")]
            rows[key] = metrics(Transcript.load(tj), args.duration)
        _write_report(out_dir, sortie, args.start, args.duration, rows)
        return

    win_wav = out_dir / "window.wav"
    win_mp4 = out_dir / "window.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(args.start), "-t", str(args.duration),
                    "-i", args.video, "-ac", "1", "-ar", "16000", str(win_wav)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(args.start), "-t", str(args.duration),
                    "-i", args.video, "-c", "copy", str(win_mp4)], check=True)
    audio_seconds = args.duration

    rows = {}
    for key in [b.strip() for b in args.backends.split(",") if b.strip()]:
        mod_name, extra = BACKENDS[key]
        print(f"\n=== {key} ({mod_name}) ===")
        try:
            fn = importlib.import_module(mod_name).transcribe
            t = timed(fn)(str(win_wav), language=args.language or None, **extra)
            t.sortie = sortie
            m = metrics(t, audio_seconds)
            rows[key] = m
            tj = out_dir / f"{key}.transcript.json"
            t.save(tj)
            print(f"  {json.dumps(m, ensure_ascii=False)}")

            if not args.no_video:
                importlib.import_module("diarize").diarize(t, win_wav, n_speakers=2)
                t.save(tj)
                subprocess.run([sys.executable, str(Path(__file__).parent / "audio_vis.py"),
                                str(win_mp4), "--transcript", str(tj), "--start", "0",
                                "--duration", str(args.duration),
                                "--out", str(out_dir / f"{key}.mp4")], check=True)
                print(f"  movie -> {out_dir / f'{key}.mp4'}")
        except Exception as e:
            rows[key] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
            print(f"  FAILED: {rows[key]['error']}")
            traceback.print_exc(limit=1)

    _write_report(out_dir, sortie, args.start, args.duration, rows)


def _write_report(out_dir, sortie, start, duration, rows):
    (out_dir / "report.json").write_text(json.dumps(
        {"sortie": sortie, "window_s": [start, start + duration], "backends": rows},
        indent=2, ensure_ascii=False))

    md = [f"# ASR backend comparison -- {sortie}",
          f"\nwindow: {start:.0f}-{start + duration:.0f}s ({duration:.0f}s of audio)\n",
          "| backend | " + " | ".join(METRIC_COLS) + " |",
          "|" + "---|" * (len(METRIC_COLS) + 1)]
    for key, m in rows.items():
        if "error" in m:
            md.append(f"| **{key}** | _{m['error']}_ |" + " |" * (len(METRIC_COLS) - 1))
        else:
            md.append(f"| **{key}** | " +
                      " | ".join(str(m.get(c, "")) for c in METRIC_COLS) + " |")
    md += ["",
           "- **words / segments / words_per_min** -- raw output volume.",
           "- **coverage** -- transcribed speech seconds / window seconds.",
           "- **mean_word_prob / low_conf_word_pct** -- only real for whisper_openai "
           "(CTC + Seamless expose no per-word confidence).",
           "- **hebrew_word_pct / latin_word_pct** -- script mix; a Hebrew-tuned model "
           "forced to `he` should be ~100% Hebrew, Seamless may surface real English.",
           "- **empty_junk_segments** -- punctuation-only / blank segments (Whisper "
           "silence artefacts).",
           "- **trigram_repetition** -- looped-output / hallucination proxy "
           "(1 - unique trigrams / total); high = the model is repeating itself.",
           "- **wall_seconds / realtime_factor** -- speed on this box."]
    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    print(f"\nreport -> {out_dir / 'report.md'}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
