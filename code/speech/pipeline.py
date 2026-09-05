"""End-to-end: sortie video -> audio -> transcript (STT + diarization +
glossary tagging), then optionally straight into every biomarker.

    python3 code/speech/pipeline.py "/home/ilan/Documents/flight school/data/s1 c2.mp4"
    python3 code/speech/pipeline.py "...s1 c2.mp4" --model medium --skip-biomarkers

Writes, under exports/audio/<sortie>/:
    <sortie>.wav (in exports/audio/, shared across steps)
    transcript.json
    glossary_review_queue.csv
    biomarkers/b01_*.json ... b15_*.json   (unless --skip-biomarkers)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diarize as diarize_mod       # noqa: E402
import glossary                     # noqa: E402
import transcribe as transcribe_mod  # noqa: E402
from schema import Transcript       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"


def run(video: str, model: str = "large-v3", speakers: int = 2,
       skip_biomarkers: bool = False, cadet_speaker: str | None = None) -> Path:
    t0 = time.time()
    print(f"[1/4] transcribing ({model}) ...")
    t = transcribe_mod.transcribe(video, model)
    print(f"      {len(t.segments)} segments, {sum(len(s.words) for s in t.segments)} words, "
          f"{t.duration:.0f}s  ({time.time() - t0:.0f}s)")

    print("[2/4] diarizing ...")
    wav = AUDIO_DIR / f"{t.sortie}.wav"
    t = diarize_mod.diarize(t, wav, n_speakers=speakers)

    print("[3/4] glossary tagging ...")
    gloss = glossary.load_glossary()
    glossary.match_transcript(t, gloss)

    out_dir = AUDIO_DIR / t.sortie
    t.save(out_dir / "transcript.json")
    q = glossary.review_queue(t, out_dir / "glossary_review_queue.csv")
    print(f"      transcript -> {out_dir / 'transcript.json'}")
    print(f"      review queue -> {q}")

    if not skip_biomarkers:
        print("[4/4] biomarkers ...")
        from run_biomarkers import run_all
        run_all(out_dir / "transcript.json", cadet_speaker=cadet_speaker)
    else:
        print("[4/4] skipped (--skip-biomarkers)")

    print(f"done in {time.time() - t0:.0f}s")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--speakers", type=int, default=2)
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--skip-biomarkers", action="store_true")
    args = ap.parse_args()
    run(args.video, args.model, args.speakers, args.skip_biomarkers, args.cadet_speaker)


if __name__ == "__main__":
    main()
