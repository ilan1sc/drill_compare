"""12.3 Action-related statements -- verbalization associated with
execution ("lowering collective", "rolling in now").

Heuristic: cadet segments matching the `action` cue lexicon
(code/biomarkers/lexicons.py) -- present-progressive / execution verbs
("lowering", "applying", "rolling in", ...).

Limitation: cue-phrase matching; silent execution (action performed
without any verbalization) is invisible to an audio-only biomarker by
definition -- pair with the flight-data drills (pfd_compare_csv.py) to
see when maneuvers happened regardless of whether they were called out.
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
        if has_cue(seg.text, "action"):
            hits.append(seg)

    duration_min = max(transcript.duration / 60.0, 1e-6)
    metrics = {
        "count": len(hits),
        "per_minute": round(len(hits) / duration_min, 3),
        "share_of_cadet_segments": round(len(hits) / total, 3) if total else 0.0,
    }
    evidence = [evidence_row(s, "action") for s in hits]
    return BiomarkerResult("b03_action_statements", metrics, evidence,
                           limitations="cue-phrase heuristic; see module docstring")


if __name__ == "__main__":
    run_cli("b03_action_statements", compute)
