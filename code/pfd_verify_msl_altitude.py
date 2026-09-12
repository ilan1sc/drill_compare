"""Plot PFD altitude (OCR) vs telemetry AltitudeMSL, at an already-known
(t_scale, t_shift) -- e.g. from pfd_compare_csv5.py's heading fit.

    telemetry : FFS1_Ownship.LHD_ESL_Flight_Model_Data.
                LHD_ESL_Flight_Model_Data.AltitudeMSL   (m -> ft)
    PFD       : `altitude` column of the OCR'd PFD csv (ft, as shown)

Known caveat: the PFD altitude readout is a split drum (a "thousands"
digit box + a separate rolling hundreds/tens box); pfd.py's OCR
concatenates whatever it reads in each as one string, so it periodically
drops the thousands digit and the reading saw-tooths by ~1000-9000 ft
(e.g. actual ~15,900 ft reads as "1500"). This plot does NOT correct that
-- it's shown as-is so the artifact is visible, not hidden.

    python3 code/pfd_verify_msl_altitude.py --pfd <pfd_csv> --data <telemetry_csv> \\
        --t-scale 0.9990 --t-shift 0.5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MSL = ("FFS1_Ownship.LHD_ESL_Flight_Model_Data."
       "LHD_ESL_Flight_Model_Data.AltitudeMSL")
M_TO_FT = 3.28084
GRID_HZ = 4.0
DT = 1.0 / GRID_HZ


def load_pfd_alt(path, window=401, jump=400):
    """The altitude readout is a split drum (thousands digit + a separate
    rolling hundreds/tens box); OCR concatenates whatever it reads in
    each, so it periodically drops/garbles the thousands digit and the
    value jumps by 1000s of ft for one to a few frames. Same cleaning
    approach as airspeed: flag samples that jump too far from a local
    moving median (a real climb/descent can't move this fast frame to
    frame) and interpolate them from the valid samples either side."""
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy(float)
    raw = pd.to_numeric(df["altitude"], errors="coerce")
    trend = raw.rolling(window, center=True, min_periods=1).median()
    dev = (raw - trend).abs()
    v = raw.where(dev <= jump)
    n_raw, n_kept = raw.notna().sum(), v.notna().sum()
    print(f"  altitude: kept {n_kept}/{n_raw} raw OCR reads within {jump}ft of the "
          f"local moving median, interpolated the rest")
    v = v.interpolate(limit_direction="both").to_numpy()
    return t, v


def load_data_msl(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_numeric(df["Recording Time"], errors="coerce") / 1000.0
    v = pd.to_numeric(df[MSL], errors="coerce") * M_TO_FT
    m = t.notna() & v.notna()
    return t[m].to_numpy(), v[m].to_numpy()


def _on_grid(t, v, grid):
    return np.interp(grid, t, v, left=np.nan, right=np.nan)


def _fit_ab(x, y):
    a, b = np.polyfit(x, y, 1)
    yp = a * x + b
    r = np.corrcoef(yp, y)[0, 1] if np.std(yp) > 0 else 0.0
    return float(a), float(b), float(r)


def fit_at(pt, pv, dt, dv, t_scale, t_shift):
    dg = np.arange(dt[0], dt[-1], DT)
    yy = _on_grid(pt * t_scale + t_shift, pv, dg)
    dgy = _on_grid(dt, dv, dg)
    ok = ~np.isnan(yy) & ~np.isnan(dgy)
    a, b, r = _fit_ab(yy[ok], dgy[ok])
    return {"t_scale": t_scale, "t_shift": t_shift, "amp_gain": a,
            "amp_offset": b, "r": r, "n": int(ok.sum())}


def render(pt, pv, dt, dv, res, label, out_png):
    wt = pt * res["t_scale"] + res["t_shift"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharey=True)
    axes[0].plot(dt, dv, color="#1f77b4", lw=0.8, label="telemetry MSL alt (ft)")
    axes[0].plot(pt, pv, color="#d62728", lw=0.6, label="PFD altitude (raw ft)")
    axes[0].set_title("altitude (ft) — as recorded, no time alignment")
    axes[0].legend(loc="upper right")

    axes[1].plot(dt, dv, color="#1f77b4", lw=0.9, label="telemetry MSL alt")
    axes[1].plot(wt, pv, color="#d62728", lw=0.7, label="PFD altitude (time-aligned)")
    axes[1].set_title(
        f"after   t: {res['t_scale']:.4f}·pfd {res['t_shift']:+.1f}s   (r = {res['r']:.3f}, n={res['n']})   "
        f"-- PFD sawtooths where OCR drops the thousands digit")
    axes[1].legend(loc="upper right")
    ymax = np.nanpercentile(dv, 99.5) * 1.3
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_ylabel("altitude (ft)")
        ax.set_ylim(0, ymax)
    axes[1].set_xlabel("telemetry time (s)")
    fig.suptitle(f"PFD altitude vs telemetry MSL altitude{(' — ' + label) if label else ''}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--t-scale", type=float, required=True)
    ap.add_argument("--t-shift", type=float, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pt, pv = load_pfd_alt(args.pfd)
    dt, dv = load_data_msl(args.data)
    print(f"PFD altitude   {len(pt)} rows, {pt[-1]:.0f}s, {np.nanmin(pv):.0f}..{np.nanmax(pv):.0f} ft")
    print(f"telemetry MSL  {len(dt)} rows, {dt[-1]:.0f}s, {np.nanmin(dv):.0f}..{np.nanmax(dv):.0f} ft")

    res = fit_at(pt, pv, dt, dv, args.t_scale, args.t_shift)
    print(f"at given fit: r={res['r']:.3f}  n={res['n']}")
    render(pt, pv, dt, dv, res, args.label, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
