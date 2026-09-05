"""12.3 Prioritization statements -- what the cadet identifies as most
important right now ("first thing, get the nose down").

Heuristic: cadet segments matching the `prioritization` cue lexicon
(code/biomarkers/lexicons.py: "first", "priority", "most important", ...).

Limitation: cue-phrase matching; sequencing words like "first" also occur
in unrelated narration ("first time I've seen that") -- spot-check
against the evidence list.
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
        if has_cue(seg.text, "prioritization"):
            hits.append(seg)

    duration_min = max(transcript.duration / 60.0, 1e-6)
    metrics = {
        "count": len(hits),
        "per_minute": round(len(hits) / duration_min, 3),
        "share_of_cadet_segments": round(len(hits) / total, 3) if total else 0.0,
    }
    evidence = [evidence_row(s, "prioritization") for s in hits]
    return BiomarkerResult("b08_prioritization_statements", metrics, evidence,
                           limitations="cue-phrase heuristic; see module docstring")


if __name__ == "__main__":
    run_cli("b08_prioritization_statements", compute)
