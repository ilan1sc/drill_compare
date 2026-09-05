"""Common transcript data shapes shared by the pipeline and every biomarker.

A Transcript is what pipeline.py writes to
exports/audio/<sortie>/transcript.json and every code/biomarkers/*.py reads.
Kept as plain dicts/dataclasses (json-serialisable) rather than a DB row so
biomarkers stay simple, dependency-free scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float                 # ASR confidence 0..1
    speaker: str | None = None  # filled in by diarize.py
    glossary_term: str | None = None   # canonical term id, if matched


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    lang: str
    speaker: str | None
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    sortie: str
    source_video: str
    duration: float
    segments: list[Segment] = field(default_factory=list)

    def words(self):
        for seg in self.segments:
            yield from seg.words

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: str | Path) -> "Transcript":
        data = json.loads(Path(path).read_text())
        segs = []
        for s in data["segments"]:
            words = [Word(**w) for w in s.get("words", [])]
            segs.append(Segment(**{**s, "words": words}))
        return Transcript(sortie=data["sortie"], source_video=data["source_video"],
                          duration=data["duration"], segments=segs)
