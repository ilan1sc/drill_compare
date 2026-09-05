"""12.3 Observation statements -- verbalization of aircraft state,
environment or relevant information ("I'm at 500 feet", "showing 60 psi").

Heuristic: cadet segments matching the `observation` cue lexicon
(code/biomarkers/lexicons.py), OR containing a glossary system/abbrev
term (e.g. altitude, torque, RA) together with a number -- a common
observation shape ("torque's at ninety") the plain cue list misses.

Limitation: cue-phrase matching, not a trained classifier; misses
observations phrased without a lexicon cue or a nearby number.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult, cadet_segments, evidence_row, run_cli  # noqa: E402
from lexicons import has_cue  # noqa: E402

NUMBER = re.compile(r"\d")


def compute(transcript, cadet_speaker=None) -> BiomarkerResult:
    hits, total = [], 0
    for seg in cadet_segments(transcript, cadet_speaker):
        total += 1
        has_glossary_num = any(w.glossary_term for w in seg.words) and NUMBER.search(seg.text)
        if has_cue(seg.text, "observation") or has_glossary_num:
            hits.append(seg)

    duration_min = max(transcript.duration / 60.0, 1e-6)
    metrics = {
        "count": len(hits),
        "per_minute": round(len(hits) / duration_min, 3),
        "share_of_cadet_segments": round(len(hits) / total, 3) if total else 0.0,
    }
    evidence = [evidence_row(s, "observation") for s in hits]
    return BiomarkerResult("b01_observation_statements", metrics, evidence,
                           limitations="cue-phrase heuristic; see module docstring")


if __name__ == "__main__":
    run_cli("b01_observation_statements", compute)
