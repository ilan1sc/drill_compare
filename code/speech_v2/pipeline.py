"""SOLUTION 2 pipeline: same stages as code/speech/pipeline.py (solution 1)
-- extract audio, transcribe, diarize, glossary-tag, run every biomarker --
but transcribing with the Hebrew-specialized ASR model
(code/speech_v2/transcribe.py, ivrit-ai/whisper-large-v3) instead of
solution 1's general-purpose openai-whisper large-v3.

Diarization, the glossary and every code/biomarkers/*.py module are
UNCHANGED and reused as-is from solution 1 -- only the ASR step differs,
so this is the fair way to compare the two: everything downstream of the
transcript is identical code.

Writes to exports/audio_v2/<sortie>/ (never exports/audio/, solution 1's
tree, which this leaves untouched) and stamps
exports/audio_v2/<sortie>/transcript.json's biomarker summary and a
SOLUTION.md marker with `solution: solution_2` / the ASR model id.

    python3 code/speech_v2/pipeline.py "/home/ilan/Documents/flight school/data/s1 c2.mp4"
"""

import argparse
import sys
import time
from pathlib import Path

# import order matters: code/speech/ also has a transcribe.py (solution 1's),
# so resolve this dir's transcribe.py *before* code/speech/ goes on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcribe as transcribe_v2   # noqa: E402  (this dir's -- the new ASR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speech"))
import diarize as diarize_mod        # noqa: E402  (solution 1's, reused)
import glossary                      # noqa: E402  (solution 1's, reused)
from schema import Transcript        # noqa: E402  (solution 1's, reused)

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio_v2"
SOLUTION = "solution_2"


def run(video: str, language: str = "he", speakers: int = 2,
       skip_biomarkers: bool = False, cadet_speaker: str | None = None) -> Path:
    t0 = time.time()
    print(f"[1/4] transcribing ({transcribe_v2.MODEL_ID}) ...")
    t = transcribe_v2.transcribe(video, language=language)

    print("[2/4] diarizing (solution 1's diarize.py, reused) ...")
    wav = AUDIO_DIR / f"{t.sortie}.wav"
    t = diarize_mod.diarize(t, wav, n_speakers=speakers)

    print("[3/4] glossary tagging (solution 1's glossary, reused) ...")
    gloss = glossary.load_glossary()
    glossary.match_transcript(t, gloss)

    out_dir = AUDIO_DIR / t.sortie
    t.save(out_dir / "transcript.json")
    q = glossary.review_queue(t, out_dir / "glossary_review_queue.csv")
    print(f"      transcript -> {out_dir / 'transcript.json'}")
    print(f"      review queue -> {q}")

    (out_dir / "SOLUTION.md").write_text(
        f"# {SOLUTION}\n\nASR model: `{transcribe_v2.MODEL_ID}` (Hebrew-specialized), "
        f"language={language or 'auto'}\n\n"
        f"Diarization / glossary / biomarkers: unchanged, reused from solution 1 "
        f"(code/speech/, code/biomarkers/).\n\n"
        f"Solution 1's output for the same sortie, if it exists, is at "
        f"`exports/audio/{t.sortie}/` -- untouched by this run.\n"
    )

    if not skip_biomarkers:
        print("[4/4] biomarkers (solution 1's modules, reused) ...")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speech"))
        import run_biomarkers
        run_biomarkers.run_all(out_dir / "transcript.json", cadet_speaker=cadet_speaker,
                               audio_dir=AUDIO_DIR, wav_path=str(wav),
                               extra_summary={"solution": SOLUTION,
                                              "asr_model": transcribe_v2.MODEL_ID})
    else:
        print("[4/4] skipped (--skip-biomarkers)")

    print(f"done in {time.time() - t0:.0f}s  [{SOLUTION}]")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--language", default="he", help='force this language, or "" to auto-detect')
    ap.add_argument("--speakers", type=int, default=2)
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--skip-biomarkers", action="store_true")
    args = ap.parse_args()
    run(args.video, args.language, args.speakers, args.skip_biomarkers, args.cadet_speaker)


if __name__ == "__main__":
    main()
