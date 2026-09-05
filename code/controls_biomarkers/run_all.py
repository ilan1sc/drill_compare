"""Run all 9 flight-control-analytics biomarkers (PRD 10) over one sortie's
data CSV and write a combined summary.

    python3 code/controls_biomarkers/run_all.py \\
        "/home/ilan/Documents/flight school/data/csv_data/1_26/CSV_3.6.26_check1/003/Elements_0.csv"
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import CONTROLS_DIR, load_flight  # noqa: E402

MODULES = [
    "controls_precision", "controls_smoothness", "controls_efficiency",
    "controls_correction_count", "controls_overshoot", "controls_correction_gain",
    "controls_anticipation", "controls_stabilization_time", "controls_aggressiveness",
]


def run_all(data_csv: str, sortie: str | None = None) -> dict:
    flight = load_flight(data_csv)
    sortie = sortie or Path(data_csv).stem
    minutes = (flight["t"][-1] - flight["t"][0]) / 60.0
    print(f"{sortie}: {minutes:.1f} min, {len(flight['t'])} samples @ {1/flight['dt']:.0f}Hz")

    summary = {"sortie": sortie, "source": data_csv, "duration_min": round(minutes, 1),
              "biomarkers": {}}
    for name in MODULES:
        mod = importlib.import_module(name)
        result = mod.compute(flight)
        result.save(sortie)
        summary["biomarkers"][name] = result.metrics
        print(f"  {name}: {json.dumps(result.metrics)[:200]}...")

    out = CONTROLS_DIR / sortie / "_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_csv")
    ap.add_argument("--sortie", default=None)
    args = ap.parse_args()
    run_all(args.data_csv, args.sortie)


if __name__ == "__main__":
    main()
