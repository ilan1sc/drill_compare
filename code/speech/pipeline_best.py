"""End-to-end with the engines that WON the bake-offs
(code/speech/transcribe_compare.py + diarize_compare.py):

  ASR / transcript : ivrit-ai/whisper-large-v3   (transcribe_ivrit_v3.py)
                     Whisper large-v3 fine-tuned on Hebrew -- best coverage,
                     no hallucination in the compare.
  diarization      : 3D-Speaker  (diarize_3dspeaker.py)
                     Alibaba CAM++ pipeline via ModelScope -- cleanest
                     2-speaker split in the compare.
  glossary         : the versioned aviation glossary (glossary.py, v2)
  biomarkers       : b01..b15 via run_biomarkers.run_all

Writes to a SEPARATE tree so it never collides with pipeline.py (sol 1)
or speech_v2 (sol 2):

    exports/audio_best/<sortie>/transcript.json
                               glossary_review_queue.csv
                               biomarkers/b01_*.json ... _summary.json

    python3 code/speech/pipeline_best.py "/home/ilan/Documents/flight school/data/s1 c2.mp4"
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diarize_3dspeaker                # noqa: E402
import glossary                         # noqa: E402
import transcribe_ivrit_v3              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"          # shared .wav lives here
AUDIO_BEST = ROOT / "exports" / "audio_best"

ENGINES = {"asr": "ivrit-ai/whisper-large-v3", "diarization": "3D-Speaker (CAM++)",
           "glossary": "aviation glossary (glossary.py)"}


def run(video: str, speakers: int | None = 2, skip_biomarkers: bool = False,
        cadet_speaker: str | None = None) -> Path:
    t0 = time.time()
    sortie = Path(video).stem

    print(f"[1/4] transcribe  ivrit-ai/whisper-large-v3 ...")
    t = transcribe_ivrit_v3.transcribe(video)          # forces language='he'
    print(f"      {len(t.segments)} segments, {sum(len(s.words) for s in t.segments)} words, "
          f"{t.duration:.0f}s  ({time.time() - t0:.0f}s)")

    print("[2/4] diarize  3D-Speaker ...")
    wav = AUDIO_DIR / f"{sortie}.wav"                  # written by transcribe's load_wav
    t = diarize_3dspeaker.diarize(t, wav, n_speakers=speakers)

    print("[3/4] glossary tag ...")
    glossary.match_transcript(t, glossary.load_glossary())

    out_dir = AUDIO_BEST / sortie
    out_dir.mkdir(parents=True, exist_ok=True)
    t.save(out_dir / "transcript.json")
    q = glossary.review_queue(t, out_dir / "glossary_review_queue.csv")
    print(f"      transcript -> {out_dir / 'transcript.json'}")
    print(f"      review queue -> {q}")

    if not skip_biomarkers:
        print("[4/4] biomarkers ...")
        from run_biomarkers import run_all
        run_all(out_dir / "transcript.json", cadet_speaker=cadet_speaker,
                wav_path=str(wav), audio_dir=AUDIO_BEST,
                extra_summary={"engines": ENGINES})
    else:
        print("[4/4] skipped")

    print(f"done {sortie} in {time.time() - t0:.0f}s")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--speakers", default="2", help="int, or 'auto' to let 3D-Speaker decide")
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--skip-biomarkers", action="store_true")
    args = ap.parse_args()
    spk = None if args.speakers == "auto" else int(args.speakers)
    run(args.video, spk, args.skip_biomarkers, args.cadet_speaker)


if __name__ == "__main__":
    main()
