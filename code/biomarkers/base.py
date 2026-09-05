"""Shared plumbing every code/biomarkers/b##_*.py module uses.

Each biomarker module exposes one function:

    compute(transcript: Transcript, **kwargs) -> BiomarkerResult

and a __main__ CLI: `python3 code/biomarkers/bNN_xxx.py transcript.json`
writes exports/audio/<sortie>/biomarkers/<name>.json and prints a summary.
run_biomarkers.py (in code/speech/) runs all of them over one transcript.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speech"))
from schema import Transcript  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"


@dataclass
class BiomarkerResult:
    name: str
    metrics: dict
    evidence: list[dict] = field(default_factory=list)   # [{t, speaker, text, tag}]
    limitations: str = ""

    def save(self, sortie: str, root: Path = AUDIO_DIR) -> Path:
        """root defaults to exports/audio/ (solution 1); pipelines writing
        elsewhere (e.g. exports/audio_v2/ for solution 2) pass their own."""
        out = root / sortie / "biomarkers" / f"{self.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
        return out


def cadet_segments(transcript: Transcript, cadet_speaker: str | None):
    """Segments to score. If cadet_speaker is None (no reliable diarization
    label supplied), scores everyone -- callers should treat results as
    'cockpit' rather than 'cadet-only' in that case."""
    for seg in transcript.segments:
        if cadet_speaker is None or seg.speaker == cadet_speaker:
            yield seg


def guess_cadet_speaker(transcript: Transcript) -> str | None:
    """Heuristic when not told which SPEAKER_xx is the cadet: the busier
    speaker (more words) on a 2-person channel is usually the cadet, who
    is talking through the maneuver while the instructor mostly listens.
    Wrong on some sorties -- pass --cadet-speaker explicitly when known."""
    counts: dict[str, int] = {}
    for w in transcript.words():
        if w.speaker:
            counts[w.speaker] = counts.get(w.speaker, 0) + 1
    return max(counts, key=counts.get) if counts else None


STATEMENT_CATEGORIES = ("observation", "intention", "action", "verification",
                        "error_correction", "prediction", "prioritization",
                        "info_sharing")


def tag_segments(transcript: Transcript, cadet_speaker: str | None):
    """category -> list of segments (from cadet_speaker, or everyone) whose
    text matched that category's cue lexicon. A segment can appear under
    several categories."""
    from lexicons import has_cue
    tags: dict[str, list] = {c: [] for c in STATEMENT_CATEGORIES}
    for seg in cadet_segments(transcript, cadet_speaker):
        for cat in STATEMENT_CATEGORIES:
            if has_cue(seg.text, cat):
                tags[cat].append(seg)
    return tags


def evidence_row(seg, tag: str = "") -> dict:
    return {"t": round(seg.start, 2), "speaker": seg.speaker, "text": seg.text, "tag": tag}


def run_cli(module_name: str, compute_fn):
    import argparse
    ap = argparse.ArgumentParser(description=f"Biomarker: {module_name}")
    ap.add_argument("transcript_json")
    ap.add_argument("--cadet-speaker", default=None,
                    help="e.g. SPEAKER_00; default: guess the busier speaker")
    args = ap.parse_args()

    t = Transcript.load(args.transcript_json)
    cadet = args.cadet_speaker or guess_cadet_speaker(t)
    result = compute_fn(t, cadet_speaker=cadet)
    out = result.save(t.sortie)
    print(f"{module_name}: {result.metrics}")
    print(f"Wrote {out}")
