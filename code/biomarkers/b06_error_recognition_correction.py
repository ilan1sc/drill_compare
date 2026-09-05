"""12.3 Error recognition and self-correction -- the cadet catching and
fixing their own mistake out loud ("wrong pedal -- correction, left
pedal").

Heuristic: cadet segments matching the `error_correction` cue lexicon
(code/biomarkers/lexicons.py: "correction", "I meant", "scratch that",
"actually", ...).

Limitation: catches *verbalized* self-correction only; silent correction
(cadet just fixes the control input without saying anything) isn't
observable from audio -- cross-check against control-input reversals in
the flight data CSV if you need the non-verbal cases too.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult, cadet_segments, evidence_row, run_cli  # noqa: E402
from lexicons import has_cue  # noqa: E402


def compute(transcript, cadet_speaker=None) -> BiomarkerResult:
    hits, total = [], 0
    for seg in cadet_segments(transcript, cadet_speaker):
        total += 1
        if has_cue(seg.text, "error_correction"):
            hits.append(seg)

    duration_min = max(transcript.duration / 60.0, 1e-6)
    metrics = {
        "count": len(hits),
        "per_minute": round(len(hits) / duration_min, 3),
        "share_of_cadet_segments": round(len(hits) / total, 3) if total else 0.0,
    }
    evidence = [evidence_row(s, "error_correction") for s in hits]
    return BiomarkerResult("b06_error_recognition_correction", metrics, evidence,
                           limitations="verbalized self-correction only; see module docstring")


if __name__ == "__main__":
    run_cli("b06_error_recognition_correction", compute)
