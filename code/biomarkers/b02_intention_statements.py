"""12.3 Intention statements -- what the cadet plans to do
("I'm going to lower the collective", "let's turn left here").

Heuristic: cadet segments matching the `intention` cue lexicon
(code/biomarkers/lexicons.py) -- future/planning phrasing ("I will",
"going to", "let's", ...).

Limitation: cue-phrase matching; misses bare imperative intentions
("turning final now" said just before acting, with no lead-in phrase --
see b03 action, which often captures those instead).
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
        if has_cue(seg.text, "intention"):
            hits.append(seg)

    duration_min = max(transcript.duration / 60.0, 1e-6)
    metrics = {
        "count": len(hits),
        "per_minute": round(len(hits) / duration_min, 3),
        "share_of_cadet_segments": round(len(hits) / total, 3) if total else 0.0,
    }
    evidence = [evidence_row(s, "intention") for s in hits]
    return BiomarkerResult("b02_intention_statements", metrics, evidence,
                           limitations="cue-phrase heuristic; see module docstring")


if __name__ == "__main__":
    run_cli("b02_intention_statements", compute)
