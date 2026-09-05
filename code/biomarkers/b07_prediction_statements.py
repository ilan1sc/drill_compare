"""12.3 Prediction / forward-thinking statements -- verbalizing what is
expected to happen next ("we should be picking up translational lift
about here").

Heuristic: cadet segments matching the `prediction` cue lexicon
(code/biomarkers/lexicons.py: "should", "expect", "about to", "will be", ...).

Limitation: cue-phrase matching; "should" is also used for obligation
("I should check fuel") not just forecasting, so some false positives are
expected -- evidence rows are included specifically so this can be
spot-checked.
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
        if has_cue(seg.text, "prediction"):
            hits.append(seg)

    duration_min = max(transcript.duration / 60.0, 1e-6)
    metrics = {
        "count": len(hits),
        "per_minute": round(len(hits) / duration_min, 3),
        "share_of_cadet_segments": round(len(hits) / total, 3) if total else 0.0,
    }
    evidence = [evidence_row(s, "prediction") for s in hits]
    return BiomarkerResult("b07_prediction_statements", metrics, evidence,
                           limitations="cue-phrase heuristic; see module docstring")


if __name__ == "__main__":
    run_cli("b07_prediction_statements", compute)
