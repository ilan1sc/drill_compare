"""12.3 Verbalization level by flight phase / workload.

Bins the cadet's words-per-minute (a rough workload/verbalization-load
proxy) into phases. Phases come from --phases-csv (columns: start,end,
label, seconds, SAME time base as the transcript i.e. the PFD video
clock) -- e.g. the drills table from code/pfd_sync/pfd_compare_csv.py, after
re-timing it to video time with code/pfd_sync/pfd_compare_csv3.py or
pfd_sync/pfd_compare_csv4.py (--no-aligned-csv off writes exactly that CSV).
Without --phases-csv, falls back to fixed 5-minute bins so the module
still runs standalone.

Limitation: words/min conflates "high workload, talking a lot" with
"low workload, chatting" -- pair with relevance (b10) to tell them apart.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult, cadet_segments, run_cli  # noqa: E402


def _phases_from_csv(path):
    phases = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            phases.append((float(row["start"]), float(row["end"]),
                           row.get("kind") or row.get("label") or "phase"))
    return phases


def _fixed_phases(duration, width=300.0):
    phases, t = [], 0.0
    while t < duration:
        phases.append((t, min(t + width, duration), f"{int(t // 60)}-{int((t + width) // 60)}min"))
        t += width
    return phases


def compute(transcript, cadet_speaker=None, phases_csv: str | None = None) -> BiomarkerResult:
    phases = _phases_from_csv(phases_csv) if phases_csv else _fixed_phases(transcript.duration)
    segs = list(cadet_segments(transcript, cadet_speaker))

    rows = []
    for start, end, label in phases:
        in_phase = [s for s in segs if start <= s.start < end]
        words = sum(len(s.words) for s in in_phase)
        mins = max((end - start) / 60.0, 1e-6)
        rows.append({
            "label": label, "start": round(start, 1), "end": round(end, 1),
            "segments": len(in_phase), "words": words,
            "words_per_min": round(words / mins, 2),
        })

    metrics = {
        "phases": len(rows),
        "phase_source": "csv" if phases_csv else "fixed_5min",
        "max_words_per_min_phase": max(rows, key=lambda r: r["words_per_min"])["label"]
                                    if rows else None,
        "min_words_per_min_phase": min(rows, key=lambda r: r["words_per_min"])["label"]
                                    if rows else None,
    }
    return BiomarkerResult("b11_verbalization_by_phase", metrics, rows,
                           limitations="workload proxy only; see module docstring")


if __name__ == "__main__":
    import argparse
    from base import Transcript, guess_cadet_speaker

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript_json")
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--phases-csv", default=None)
    args = ap.parse_args()

    t = Transcript.load(args.transcript_json)
    cadet = args.cadet_speaker or guess_cadet_speaker(t)
    result = compute(t, cadet_speaker=cadet, phases_csv=args.phases_csv)
    out = result.save(t.sortie)
    print(f"b11_verbalization_by_phase: {result.metrics}")
    print(f"Wrote {out}")
