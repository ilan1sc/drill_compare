"""12.3 Plan clarity and communication structure.

For every intention statement (b02's tags), scores three cheap structure
proxies and averages them into a 0..1 clarity_score per statement:
  - names_object   : contains a glossary term (a concrete system/maneuver,
                     not just "I'm gonna do it")
  - sequenced      : uses a sequencing word (first/then/next/after/before)
  - followed_through: an action-tagged segment (b03) occurs within
                     --follow-window seconds afterward

Limitation: these are structural proxies, not a real clarity/coherence
model; a long rambling intention with a glossary word in it still scores
"names_object". Use the evidence list to sanity-check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult, run_cli, tag_segments  # noqa: E402

SEQ_WORDS = ["first", "then", "next", "after", "before", "once", "קודם", "אחר כך", "אז"]


def compute(transcript, cadet_speaker=None, follow_window: float = 20.0) -> BiomarkerResult:
    tags = tag_segments(transcript, cadet_speaker)
    intentions, actions = tags["intention"], tags["action"]
    action_starts = sorted(a.start for a in actions)

    rows, scores = [], []
    for seg in intentions:
        names_object = any(w.glossary_term for w in seg.words)
        sequenced = any(f" {sw} " in f" {seg.text.lower()} " for sw in SEQ_WORDS)
        followed = any(seg.end <= a <= seg.end + follow_window for a in action_starts)
        score = sum([names_object, sequenced, followed]) / 3.0
        scores.append(score)
        rows.append({
            "t": round(seg.start, 2), "text": seg.text, "clarity_score": round(score, 2),
            "names_object": names_object, "sequenced": sequenced, "followed_through": followed,
        })

    metrics = {
        "intention_statements": len(intentions),
        "avg_clarity_score": round(sum(scores) / len(scores), 3) if scores else None,
        "follow_through_rate": round(sum(r["followed_through"] for r in rows) / len(rows), 3)
                               if rows else None,
        "sequencing_rate": round(sum(r["sequenced"] for r in rows) / len(rows), 3) if rows else None,
    }
    return BiomarkerResult("b12_plan_clarity", metrics, rows,
                           limitations="structural proxies, not a coherence model; "
                                       "see module docstring")


if __name__ == "__main__":
    run_cli("b12_plan_clarity", compute)
