"""12.3 Information sharing / CRM -- timely communication that creates
shared situational awareness ("traffic, two o'clock", "you have controls").

Heuristic: segments from BOTH speakers (CRM is a two-way behavior, so this
one does not restrict to cadet_speaker) matching the `info_sharing` cue
lexicon (code/biomarkers/lexicons.py). Also reports the split between
cadet-initiated and instructor-initiated info sharing when diarization
labels are available.

Limitation: cue-phrase matching; a lot of genuine CRM (e.g. simply
narrating a scan result the other person didn't have) doesn't use one of
these stock phrases and will be missed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult, evidence_row, run_cli  # noqa: E402
from lexicons import has_cue  # noqa: E402


def compute(transcript, cadet_speaker=None) -> BiomarkerResult:
    hits = [seg for seg in transcript.segments if has_cue(seg.text, "info_sharing")]
    by_speaker = {}
    for s in hits:
        by_speaker[s.speaker or "?"] = by_speaker.get(s.speaker or "?", 0) + 1

    duration_min = max(transcript.duration / 60.0, 1e-6)
    metrics = {
        "count": len(hits),
        "per_minute": round(len(hits) / duration_min, 3),
        "by_speaker": by_speaker,
        "cadet_initiated": by_speaker.get(cadet_speaker, 0) if cadet_speaker else None,
    }
    evidence = [evidence_row(s, "info_sharing") for s in hits]
    return BiomarkerResult("b09_information_sharing_crm", metrics, evidence,
                           limitations="cue-phrase heuristic, both speakers; see module docstring")


if __name__ == "__main__":
    run_cli("b09_information_sharing_crm", compute)
