"""11.1 Change in scan strategy across sorties.

Compares a SEQUENCE of sorties for the same seat (same cadet, assuming
seat assignment is consistent -- check that before trusting this) on:

    gaze_entropy.entropy_norm            spread of attention
    scan_path_regularity.transition_entropy_norm   scan-pattern regularity
    fixation_duration_frequency.mean_duration_s    dwell style
    time_inside_vs_outside_panel.inside_panel_share

Each already-computed per-sortie biomarker JSON is read (run the others
first, e.g. via run_all.py) and the trend is the slope of a linear fit
against sortie index, same convention as
code/biomarkers/b16_learning_progression.py.

Limitation: AOI-clustering is independent per sortie (KMeans re-fit every
time), so "AOI_3" in one sortie is not the same physical direction as
"AOI_3" in another -- only the AOI-INDEPENDENT numbers above (entropy,
duration, inside/outside share) are safe to compare directly across
sorties; a real cross-sortie AOI comparison needs the SAME calibrated
labels applied to each (aoi_labels.py), which this does not assume.

Usage:
    python3 code/gaze/scan_strategy_change.py left "s1 c2" "s1 c4" "s1 c6"
(sortie names oldest-first; must all share the given seat)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import GAZE_DIR, Result  # noqa: E402

METRICS = {
    "gaze_entropy": ("entropy_norm", None),
    "scan_path_regularity": ("transition_entropy_norm", None),
    "fixation_duration_frequency": ("mean_duration_s", None),
    "time_inside_vs_outside_panel": ("inside_panel_share", None),
}


def _load(sortie, seat, module, key):
    p = GAZE_DIR / sortie / seat / f"{module}.json"
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


def compute(seat: str, sorties: list[str]) -> Result:
    series, trend = {}, {}
    for module, (key, _) in METRICS.items():
        vals = [_load(s, seat, module, key) for s in sorties]
        series[f"{module}.{key}"] = vals
        m = _slope(vals)
        trend[f"{module}.{key}"] = {"slope_per_sortie": round(m, 4)} if m is not None else None

    metrics = {"seat": seat, "sorties": sorties, "trend": trend}
    evidence = [{"sortie": s, **{k: v[i] for k, v in series.items()}}
               for i, s in enumerate(sorties)]
    return Result("scan_strategy_change", metrics, evidence,
                 limitations="only AOI-independent metrics compared across sorties; "
                             "see module docstring")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seat", choices=["left", "right"])
    ap.add_argument("sorties", nargs="+", help="sortie names, oldest first")
    args = ap.parse_args()

    result = compute(args.seat, args.sorties)
    out = GAZE_DIR / "progression" / f"{args.seat}_{'__'.join(args.sorties)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"name": result.name, "metrics": result.metrics, "evidence": result.evidence,
         "limitations": result.limitations}, indent=2, ensure_ascii=False))
    print(json.dumps(result.metrics["trend"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
