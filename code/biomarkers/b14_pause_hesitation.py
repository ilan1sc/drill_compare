"""12.3 Pause/hesitation patterns, interpreted in scenario context rather
than through universal thresholds.

Per PRD wording: no fixed "300ms = hesitation" cutoff. Instead this
computes the cadet's own inter-word gap distribution for THIS sortie (or
per-phase if --phases-csv is given, e.g. the re-timed drills table from
pfd_compare_csv.py/csv3.py/csv4.py) and flags a gap as a hesitation only
if it is an outlier relative to that local baseline: gap > median +
k * MAD (k = --k, default 3; MAD = median absolute deviation, robust to
the outliers themselves). Gaps longer than --turn-cap (default 8s) are
treated as a turn/topic break, not a hesitation, and excluded from both
the baseline and the flags.

Limitation: still a purely acoustic-timing signal -- says nothing about
*why* the cadet paused (task load vs. genuine uncertainty vs. simply
listening to the instructor).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult, cadet_segments, run_cli  # noqa: E402


def _gaps(segs, turn_cap, min_gap=0.02):
    """Inter-word silences, as (before, after, gap_s). Whisper's word
    timestamps often abut exactly (gap == 0) between co-articulated words
    within a phrase -- that's not silence, so those are dropped (min_gap)
    along with anything past turn_cap (a turn/topic break, not a pause)."""
    words = [w for seg in segs for w in seg.words]
    words.sort(key=lambda w: w.start)
    out = []
    for i in range(1, len(words)):
        g = words[i].start - words[i - 1].end
        if min_gap < g <= turn_cap:
            out.append((words[i - 1], words[i], g))
    return out


def _baseline(gaps, k):
    if not gaps:
        return None, None, None
    vals = sorted(g for *_, g in gaps)
    n = len(vals)
    median = vals[n // 2]
    mad = sorted(abs(v - median) for v in vals)[n // 2] or 1e-6
    return median, mad, median + k * mad


def compute(transcript, cadet_speaker=None, turn_cap: float = 8.0, k: float = 3.0,
           phases_csv: str | None = None) -> BiomarkerResult:
    segs = list(cadet_segments(transcript, cadet_speaker))
    gaps = _gaps(segs, turn_cap)
    median, mad, thresh = _baseline(gaps, k)

    flagged = []
    if thresh is not None:
        for before, after, g in gaps:
            if g > thresh:
                flagged.append({"t": round(before.end, 2), "gap_s": round(g, 2),
                                "before": before.text, "after": after.text})

    duration_min = max(transcript.duration / 60.0, 1e-6)
    metrics = {
        "gaps_measured": len(gaps),
        "baseline_median_s": round(median, 3) if median is not None else None,
        "baseline_mad_s": round(mad, 3) if mad is not None else None,
        "hesitation_threshold_s": round(thresh, 3) if thresh is not None else None,
        "hesitations": len(flagged),
        "hesitations_per_min": round(len(flagged) / duration_min, 3),
    }

    if phases_csv:
        phases = []
        with open(phases_csv) as fh:
            for row in csv.DictReader(fh):
                phases.append((float(row["start"]), float(row["end"]),
                              row.get("kind") or row.get("label") or "phase"))
        by_phase = []
        for start, end, label in phases:
            pg = [(b, a, g) for b, a, g in gaps if start <= b.start < end]
            _, _, pthresh = _baseline(pg, k)
            pflag = sum(1 for *_, g in pg if pthresh is not None and g > pthresh)
            by_phase.append({"label": label, "gaps": len(pg),
                             "threshold_s": round(pthresh, 3) if pthresh else None,
                             "hesitations": pflag})
        metrics["by_phase"] = by_phase

    return BiomarkerResult("b14_pause_hesitation", metrics, flagged,
                           limitations="acoustic timing only; see module docstring")


if __name__ == "__main__":
    import argparse
    from base import Transcript, guess_cadet_speaker

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript_json")
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--phases-csv", default=None)
    ap.add_argument("--k", type=float, default=3.0)
    args = ap.parse_args()

    t = Transcript.load(args.transcript_json)
    cadet = args.cadet_speaker or guess_cadet_speaker(t)
    result = compute(t, cadet_speaker=cadet, k=args.k, phases_csv=args.phases_csv)
    out = result.save(t.sortie)
    print(f"b14_pause_hesitation: {result.metrics}")
    print(f"Wrote {out}")
