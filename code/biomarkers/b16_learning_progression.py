"""12.3 Learning / progression -- changes in communication structure and
loop closure after feedback and across sorties.

Unlike b01-b15 (one sortie in, one result out) this one compares the
*already-computed* biomarker outputs of several sorties for the same
cadet, in chronological order, and reports the trend of the headline
number from each:

    b05 loop_complete_rate      b10 relevance_share_words
    b12 avg_clarity_score       b13 disfluency_rate (lower is better)
    b14 hesitations_per_min (lower is better)

Trend = sign of the slope of a simple linear fit against sortie index.
"improving" / "declining" is oriented so that, e.g., a falling
disfluency_rate still reads as "improving".

Usage:
    python3 code/biomarkers/b16_learning_progression.py \\
        "s1 c7 pfd rothman roitkof" "s1 c8 pdf sorek lifman" ...
(sortie names = the transcript's `sortie` field / the folder name under
exports/audio/, given oldest-first)

Limitation: needs each sortie's b05/b10/b12/b13/b14 to have already been
run (code/speech/run_biomarkers.py); says nothing about *why* a metric
moved (could be a harder maneuver set, not a skill change).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult  # noqa: E402

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "exports" / "audio"

METRICS = {
    "b05_oiav_loop": ("loop_complete_rate", "higher_better"),
    "b10_communication_relevance": ("relevance_share_words", "higher_better"),
    "b12_plan_clarity": ("avg_clarity_score", "higher_better"),
    "b13_disfluencies_restarts": ("disfluency_rate", "lower_better"),
    "b14_pause_hesitation": ("hesitations_per_min", "lower_better"),
}


def _load_metric(sortie: str, module: str, key: str):
    p = AUDIO_DIR / sortie / "biomarkers" / f"{module}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["metrics"].get(key)


def _slope(ys):
    xs = list(range(len(ys)))
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts) or 1e-9
    return num / den


def compute(sorties: list[str]) -> BiomarkerResult:
    series, trend = {}, {}
    for module, (key, direction) in METRICS.items():
        vals = [_load_metric(s, module, key) for s in sorties]
        series[f"{module}.{key}"] = vals
        m = _slope(vals)
        if m is None:
            trend[f"{module}.{key}"] = None
        else:
            improving = (m > 0) == (direction == "higher_better")
            trend[f"{module}.{key}"] = {
                "slope_per_sortie": round(m, 4),
                "direction": "improving" if improving and abs(m) > 1e-6 else
                            ("declining" if abs(m) > 1e-6 else "flat"),
            }

    metrics = {"sorties": sorties, "trend": trend}
    evidence = [{"sortie": s, **{k: v[i] for k, v in series.items()}}
               for i, s in enumerate(sorties)]
    return BiomarkerResult("b16_learning_progression", metrics, evidence,
                           limitations="requires prior per-sortie biomarker runs; "
                                       "see module docstring")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sorties", nargs="+", help="sortie names, oldest first")
    args = ap.parse_args()

    result = compute(args.sorties)
    out = AUDIO_DIR / "progression" / f"{'__'.join(args.sorties)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"name": result.name, "metrics": result.metrics, "evidence": result.evidence,
         "limitations": result.limitations}, indent=2, ensure_ascii=False))
    print(json.dumps(result.metrics["trend"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
