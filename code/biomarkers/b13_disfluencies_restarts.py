"""12.3 Restarts, self-repairs and disfluencies (speech mechanics -- not
to be confused with b06, which is about catching a *flying* error).

Three cheap word-level signals from ASR word timestamps:
  - fillers        : "um/uh/er/..." (code/biomarkers/lexicons.py)
  - immediate repeats: the same word said twice in a row ("the the fuel")
  - restarts        : the same word repeated with a pause (> --gap s)
                     between the two, i.e. cadet stalled and re-started
                     the word/phrase

Limitation: word-level pattern matching over whisper output; whisper
itself sometimes "cleans up" disfluencies during decoding, so these
counts are a floor, not the true rate. No attempt at deeper repair
grammar (e.g. "turn left -- I mean right").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult, cadet_segments, run_cli  # noqa: E402
from lexicons import CUES  # noqa: E402

FILLERS = {w.lower() for w in (*CUES["disfluency_fillers"]["en"], *CUES["disfluency_fillers"]["he"])}


def compute(transcript, cadet_speaker=None, gap: float = 0.3) -> BiomarkerResult:
    fillers, repeats, restarts = [], [], []
    total_words = 0
    for seg in cadet_segments(transcript, cadet_speaker):
        words = seg.words
        total_words += len(words)
        for i, w in enumerate(words):
            if w.text.strip(".,!?").lower() in FILLERS:
                fillers.append({"t": round(w.start, 2), "text": w.text})
            if i > 0 and words[i - 1].text.strip(".,!?").lower() == w.text.strip(".,!?").lower() \
                    and w.text.strip(".,!?"):
                gap_s = w.start - words[i - 1].end
                row = {"t": round(w.start, 2), "text": w.text, "gap_s": round(gap_s, 2)}
                (restarts if gap_s > gap else repeats).append(row)

    n = max(total_words, 1)
    metrics = {
        "total_words": total_words,
        "fillers": len(fillers), "filler_rate": round(len(fillers) / n, 4),
        "immediate_repeats": len(repeats), "repeat_rate": round(len(repeats) / n, 4),
        "restarts_with_pause": len(restarts), "restart_rate": round(len(restarts) / n, 4),
        "disfluency_rate": round((len(fillers) + len(repeats) + len(restarts)) / n, 4),
    }
    evidence = ([{"kind": "filler", **e} for e in fillers]
               + [{"kind": "repeat", **e} for e in repeats]
               + [{"kind": "restart", **e} for e in restarts])
    return BiomarkerResult("b13_disfluencies_restarts", metrics, evidence,
                           limitations="word-level pattern matching, floor estimate; "
                                       "see module docstring")


if __name__ == "__main__":
    run_cli("b13_disfluencies_restarts", compute)
