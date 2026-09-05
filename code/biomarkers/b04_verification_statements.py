"""12.3 Verification statements -- confirmation that the intended result
occurred ("confirmed, on speed", "torque's good", "roger, checks out").

Heuristic: cadet segments matching the `verification` cue lexicon
(code/biomarkers/lexicons.py).

Limitation: cue-phrase matching; a bare restatement of a number without a
confirmation word ("...ninety") is scored as observation (b01), not
verification, even if it functionally verifies a prior target.
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
        if has_cue(seg.text, "verification"):
            hits.append(seg)

    duration_min = max(transcript.duration / 60.0, 1e-6)
    metrics = {
        "count": len(hits),
        "per_minute": round(len(hits) / duration_min, 3),
        "share_of_cadet_segments": round(len(hits) / total, 3) if total else 0.0,
    }
    evidence = [evidence_row(s, "verification") for s in hits]
    return BiomarkerResult("b04_verification_statements", metrics, evidence,
                           limitations="cue-phrase heuristic; see module docstring")


if __name__ == "__main__":
    run_cli("b04_verification_statements", compute)
