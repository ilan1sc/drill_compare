"""12.3 Communication relevance -- useful, mission-related communication
rather than raw talkativeness.

Heuristic: a cadet segment counts as "relevant" if it matches any of the
substantive categories (observation/intention/action/verification/
prediction/prioritization/info_sharing) or contains a glossary term
(code/speech/glossary.py). Reports relevance as a share of segments AND
as a share of words, plus raw talkativeness (words/min) so "talks a lot"
and "talks usefully" can be told apart.

Limitation: relevance is inferred from the same cue lexicons as the other
biomarkers, so anything they miss is scored irrelevant here too --
treat "relevance_share" as a lower bound.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import (BiomarkerResult, cadet_segments, evidence_row,  # noqa: E402
                  run_cli, STATEMENT_CATEGORIES)
from lexicons import has_cue  # noqa: E402


def _is_relevant(seg) -> bool:
    return (any(has_cue(seg.text, c) for c in STATEMENT_CATEGORIES)
            or any(w.glossary_term for w in seg.words))


def compute(transcript, cadet_speaker=None) -> BiomarkerResult:
    segs = list(cadet_segments(transcript, cadet_speaker))
    relevant = [s for s in segs if _is_relevant(s)]
    total_words = sum(len(s.words) for s in segs)
    relevant_words = sum(len(s.words) for s in relevant)
    duration_min = max(transcript.duration / 60.0, 1e-6)

    metrics = {
        "segments_total": len(segs),
        "segments_relevant": len(relevant),
        "relevance_share_segments": round(len(relevant) / len(segs), 3) if segs else 0.0,
        "relevance_share_words": round(relevant_words / total_words, 3) if total_words else 0.0,
        "talkativeness_words_per_min": round(total_words / duration_min, 2),
        "relevant_words_per_min": round(relevant_words / duration_min, 2),
    }
    evidence = [evidence_row(s, "relevant") for s in relevant]
    return BiomarkerResult("b10_communication_relevance", metrics, evidence,
                           limitations="relevance = matches other biomarkers' cue lexicons "
                                       "or a glossary term; see module docstring")


if __name__ == "__main__":
    run_cli("b10_communication_relevance", compute)
