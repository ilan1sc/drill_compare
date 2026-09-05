"""Run every single-sortie biomarker (b01..b15) over one transcript.json
and write a combined summary next to the per-biomarker JSON files.

    python3 code/speech/run_biomarkers.py exports/audio/<sortie>/transcript.json
    python3 code/speech/run_biomarkers.py ...transcript.json --phases-csv drills_video_time.csv

b16 (learning_progression) compares MULTIPLE sorties and is run
separately: code/biomarkers/b16_learning_progression.py sortie1 sortie2 ...
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "biomarkers"))
from schema import Transcript  # noqa: E402
from base import guess_cadet_speaker  # noqa: E402

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "exports" / "audio"

MODULES = [
    "b01_observation_statements", "b02_intention_statements", "b03_action_statements",
    "b04_verification_statements", "b05_oiav_loop", "b06_error_recognition_correction",
    "b07_prediction_statements", "b08_prioritization_statements",
    "b09_information_sharing_crm", "b10_communication_relevance",
    "b11_verbalization_by_phase", "b12_plan_clarity", "b13_disfluencies_restarts",
    "b14_pause_hesitation", "b15_acoustic_candidates",
]

# modules whose compute() takes extra kwargs beyond (transcript, cadet_speaker)
EXTRA_KWARGS = {
    "b11_verbalization_by_phase": "phases_csv",
    "b14_pause_hesitation": "phases_csv",
    "b15_acoustic_candidates": "wav_path",
}


def run_all(transcript_json: str | Path, cadet_speaker: str | None = None,
           phases_csv: str | None = None, wav_path: str | None = None,
           audio_dir: Path = AUDIO_DIR, extra_summary: dict | None = None) -> dict:
    """audio_dir: where biomarkers/ and _summary.json land -- defaults to
    exports/audio/ (solution 1); pass exports/audio_v2/ etc. for other
    pipelines so they never write into solution 1's tree."""
    t = Transcript.load(transcript_json)
    cadet = cadet_speaker or guess_cadet_speaker(t)
    print(f"cadet speaker: {cadet}")

    summary = {"sortie": t.sortie, "cadet_speaker": cadet, **(extra_summary or {}),
              "biomarkers": {}}
    for name in MODULES:
        mod = importlib.import_module(name)
        kwargs = {"cadet_speaker": cadet}
        extra = EXTRA_KWARGS.get(name)
        if extra == "phases_csv" and phases_csv:
            kwargs["phases_csv"] = phases_csv
        if extra == "wav_path" and wav_path:
            kwargs["wav_path"] = wav_path
        result = mod.compute(t, **kwargs)
        result.save(t.sortie, root=audio_dir)
        summary["biomarkers"][name] = result.metrics
        print(f"  {name}: {result.metrics}")

    out = audio_dir / t.sortie / "biomarkers" / "_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript_json")
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--phases-csv", default=None,
                    help="drills table in video time, for b11/b14 phase binning")
    ap.add_argument("--wav", default=None)
    args = ap.parse_args()
    run_all(args.transcript_json, args.cadet_speaker, args.phases_csv, args.wav)


if __name__ == "__main__":
    main()
