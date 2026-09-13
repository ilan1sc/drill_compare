#!/usr/bin/env python3
"""Clean a PFD-OCR csv (from pfd.py) into a physically-coherent file.

This is a helicopter: airspeed, altitude, radar altitude and torque can't
jump the way single-frame OCR misreads do -- a rolling-drum readout
(altitude, airspeed) either reads clean or drops to 0 / a garbled
stacked-digit value for a run of frames mid-roll.

Method (picked from a real data-science pass on sortie 003,
2026-09-12): per field,

  1. range-clip to the physically possible band (same lo/hi the OCR ROI
     already uses),
  2. a frame-to-frame physical-rate gate: reject a reading that implies
     moving faster than this aircraft's real max rate (measured from the
     sortie's own telemetry -- see FIELDS below), compared against the
     last ACCEPTED value so one bad read can never anchor the next
     comparison,
  3. a wide-window Hampel filter (median-of-neighbourhood outlier
     rejection): the rate gate alone can't see a run that's *stuck* at one
     wrong value (each step's delta is 0, so it always passes step 2) --
     altitude's OCR misread runs were measured up to 135 consecutive
     frames (~4.5s at 30fps) long, so the window has to outlast that,
  4. linear interpolation through whatever got rejected, from the
     nearest valid samples either side.

Real telemetry max rates measured on sortie 003 (informing frame_gate,
with a ~10-20x margin so genuine aggressive maneuvering never gets
gated): airspeed 7.4 kt/s, altitude 49 ft/s, heading 77.5 deg/s. RA and
torque were already comparatively clean in the raw OCR.

    python3 code/pfd_sync/data_pfd_cleaner.py "exports/pfd/003_F2-GARMIN_PFD COCKPIT_pfd.csv"
    python3 code/pfd_sync/data_pfd_cleaner.py <in.csv> --out <out.csv>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# frame_gate: max plausible change per frame (same units as the column).
# hampel_window_s / hampel_gate: for the stuck-run check -- window long
# enough to outlast a multi-second bad run, gate = max plausible drift
# across that window. circular: wrap-aware (heading only).
FIELDS = {
    # lo/hi taken from the simulator's own valid range for each channel
    # (per the user, 2026-09-13), not the OCR ROI's lo/hi -- much tighter
    # than the ROI bounds, so this alone rejects most of the huge
    # drum-misread spikes before Hampel/rate-gate even run.
    "airspeed": dict(lo=20, hi=150,  frame_gate=5.0,  hampel_window_s=60.0, hampel_gate=20.0,  circular=False),
    "altitude": dict(lo=0,  hi=2500, frame_gate=30.0, hampel_window_s=60.0, hampel_gate=400.0, circular=False),
    "heading":  dict(lo=0,  hi=360,  frame_gate=5.0,  hampel_window_s=30.0, hampel_gate=30.0,  circular=True),
    "ra":       dict(lo=0,  hi=2500, frame_gate=2.0,  hampel_window_s=30.0, hampel_gate=15.0,  circular=False),
    "pi":       dict(lo=0,  hi=130,  frame_gate=3.0,  hampel_window_s=30.0, hampel_gate=8.0,   circular=False),
}


def _circ_delta(a, b):
    """Smallest signed angular difference a-b, in [-180, 180]."""
    return (a - b + 180) % 360 - 180


def clean_channel(t: np.ndarray, raw: pd.Series, cfg: dict):
    """Order matters: a frame-to-frame gate that chains to the last
    ACCEPTED value is fooled by a gradual multi-second drift to a wrong
    reading (each step individually small enough to pass) -- observed on
    altitude as a smooth-looking sawtooth ramp down to 0 and back over
    several seconds, not a single spike. So: Hampel first, computed
    straight off the range-clipped RAW series (nothing chained, so it
    can't be poisoned by its own prior decisions), wide enough to
    outlast that. The frame-to-frame gate runs second, as a light
    cleanup pass on what Hampel already left mostly clean."""
    dt = float(np.median(np.diff(t)))
    n_raw = raw.notna().sum()

    s = raw.where((raw >= cfg["lo"]) & (raw <= cfg["hi"]))

    # pass 1: wide-window Hampel vs the raw series' own rolling median --
    # skipped for heading, which isn't a rolling drum and didn't need it.
    if not cfg["circular"]:
        window = max(5, int(round(cfg["hampel_window_s"] / dt)))
        trend = s.rolling(window, center=True, min_periods=5).median()
        dev = (s - trend).abs()
        s = s.where(dev <= cfg["hampel_gate"])

    # pass 2: frame-to-frame rate gate vs the last ACCEPTED sample --
    # cheap cleanup for whatever short spikes Hampel's window missed
    out = s.to_numpy(dtype=float).copy()
    tt = t
    last_good, last_t = None, None
    for i in range(len(out)):
        if np.isnan(out[i]):
            continue
        if last_good is None:
            last_good, last_t = out[i], tt[i]
            continue
        elapsed = max(tt[i] - last_t, dt)
        delta = abs(_circ_delta(out[i], last_good)) if cfg["circular"] else abs(out[i] - last_good)
        if delta > cfg["frame_gate"] * (elapsed / dt):
            out[i] = np.nan
        else:
            last_good, last_t = out[i], tt[i]

    s = pd.Series(out, index=raw.index)
    n_kept = s.notna().sum()
    s = s.interpolate(limit_direction="both")
    return s.to_numpy(), int(n_kept), int(n_raw)


def clean_file(in_csv: Path, out_csv: Path) -> dict:
    df = pd.read_csv(in_csv)
    t = df["time_s"].to_numpy(float)
    report = {}
    print(f"{in_csv}  ({len(df)} rows)")
    for field, cfg in FIELDS.items():
        if field not in df.columns:
            continue
        cleaned, n_kept, n_raw = clean_channel(t, pd.to_numeric(df[field], errors="coerce"), cfg)
        df[field] = cleaned
        report[field] = {"kept": n_kept, "raw": n_raw}
        pct = 100 * n_kept / max(n_raw, 1)
        print(f"  {field:10s} kept {n_kept:6d}/{n_raw:6d} raw OCR reads ({pct:5.1f}%)")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pfd_csv")
    ap.add_argument("--out", help="output CSV path (default: <input>_clean.csv next to it)")
    args = ap.parse_args()
    in_csv = Path(args.pfd_csv)
    out_csv = Path(args.out) if args.out else in_csv.with_name(in_csv.stem + "_clean.csv")
    clean_file(in_csv, out_csv)


if __name__ == "__main__":
    main()
