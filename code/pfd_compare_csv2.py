"""Fit the PFD video CSV to the data CSV on ALTITUDE, allowing the video
timeline to be both stretched and shifted.

A plain time offset did not line the two files up (see pfd_compare_csv.py).
This one also lets the video clock run fast or slow: it searches for

        data_time  ~=  scale * pfd_time + shift

that best matches the two altitude traces.

Steps
-----
1. Read altitude from each file, print min / max.
     PFD : `altitude` column (ft), light spike clean.
     data: AltitudeMSL (m) -> ft.
2. Scale each altitude trace onto 0..1 using its own min..max so the
   shapes are comparable regardless of units / offset.
3. For every trial `scale`, warp the PFD time axis, cross-correlate the
   warped PFD altitude against the data altitude to get the best `shift`
   and a correlation score in one shot; keep the best (scale, shift).
4. Refine `scale` around the best value.
5. Print the fit and write a before/after Plotly HTML (altitude only).

Usage
-----
    python3 code/pfd_compare_csv2.py
    python3 code/pfd_compare_csv2.py --scale-min 0.8 --scale-max 1.3
    python3 code/pfd_compare_csv2.py --pfd X_pfd.csv --data Y/Elements_0.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
DEF_PFD = ROOT / "exports" / "pfd" / "s1 c7 pfd rothman roitkof_pfd.csv"
DEF_DATA = ("/home/ilan/Documents/flight school/data/csv_data/1_26/"
            "CSV_3.6.26_check1/007/Elements_0.csv")
FM = "FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data."
M2FT = 3.28084
DT = 0.5                     # s, grid the traces are compared on


def load_pfd_alt(path):
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy(float)
    a = pd.to_numeric(df["altitude"], errors="coerce").clip(0, 2500)
    # OCR spikes: reject anything far from a wide running median, then smooth
    med = a.rolling(91, center=True, min_periods=1).median()
    a = a.where((a - med).abs() <= 200, med)
    a = a.rolling(31, center=True, min_periods=1).median() \
         .interpolate(limit_direction="both").to_numpy()
    return t, a


def load_data_alt(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_numeric(df["Recording Time"], errors="coerce") / 1000.0
    a = pd.to_numeric(df[FM + "AltitudeMSL"], errors="coerce") * M2FT
    m = t.notna() & a.notna()
    return t[m].to_numpy(), a[m].to_numpy()


def unit_scale(y):
    lo, hi = np.min(y), np.max(y)
    return (y - lo) / (hi - lo) if hi > lo else y * 0.0, lo, hi


def resample(t, y, grid):
    return np.interp(grid, t, y, left=np.nan, right=np.nan)


def best_shift(pt, py, gd, dy):
    """Cross-correlate warped PFD altitude vs data altitude on grid gd.
    Returns (shift_seconds, pearson_r) for the peak."""
    gp = np.arange(pt[0], pt[-1], DT)
    yp = np.interp(gp, pt, py)
    a = yp - yp.mean()
    b = dy - np.nanmean(dy)
    b = np.nan_to_num(b)
    c = signal.correlate(b, a, mode="full")
    lags = signal.correlation_lags(len(b), len(a), mode="full")
    shifts = gd[0] - gp[0] + lags * DT
    k = int(np.argmax(c))
    shift = float(shifts[k])
    # pearson r at that shift, over the real overlap
    warp = pt + shift
    yy = resample(warp, py, gd)
    ok = ~np.isnan(yy) & ~np.isnan(dy)
    if ok.sum() < 50:
        return shift, 0.0
    r = np.corrcoef(yy[ok], dy[ok])[0, 1]
    return shift, float(r)


def fit(pt, pa, dt, da, scale_min, scale_max, n_coarse=81):
    pan, plo, phi = unit_scale(pa)
    dan, dlo, dhi = unit_scale(da)
    gd = np.arange(dt[0], dt[-1], DT)
    dy = np.interp(gd, dt, dan)

    def scan(scales):
        out = []
        for s in scales:
            sh, r = best_shift(pt * s, pan, gd, dy)
            out.append((r, s, sh))
        return max(out)

    r, s, sh = scan(np.linspace(scale_min, scale_max, n_coarse))
    step = (scale_max - scale_min) / (n_coarse - 1)
    r, s, sh = scan(np.linspace(s - step, s + step, 41))
    return {"scale": s, "shift": sh, "r": r,
            "pfd_lo": plo, "pfd_hi": phi, "data_lo": dlo, "data_hi": dhi}


def build_figure(pt, pa, dt, da, res):
    pan = unit_scale(pa)[0]
    dan = unit_scale(da)[0]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
        subplot_titles=[
            "altitude, normalised — as recorded",
            f"after  data_t = {res['scale']:.4f} * pfd_t {res['shift']:+.1f}  "
            f"(r = {res['r']:.3f})"])
    for row, warp in ((1, pt), (2, pt * res["scale"] + res["shift"])):
        fig.add_trace(go.Scatter(x=dt, y=dan, line=dict(color="#1f77b4", width=1),
                                 name="data", legendgroup="d",
                                 showlegend=(row == 1)), row=row, col=1)
        fig.add_trace(go.Scatter(x=warp, y=pan, line=dict(color="#d62728", width=1),
                                 name="PFD", legendgroup="p",
                                 showlegend=(row == 1)), row=row, col=1)
    fig.update_layout(height=700, width=1400, template="plotly_white",
                      legend=dict(orientation="h"),
                      title="PFD vs data altitude — stretch + shift fit")
    fig.update_xaxes(title_text="data time (s)")
    fig.update_yaxes(title_text="alt (0–1)")
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd", default=str(DEF_PFD))
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--scale-min", type=float, default=0.7)
    ap.add_argument("--scale-max", type=float, default=1.6)
    ap.add_argument("--out", help="output HTML path")
    args = ap.parse_args()

    pt, pa = load_pfd_alt(Path(args.pfd))
    dt, da = load_data_alt(args.data)

    print(f"PFD  {args.pfd}")
    print(f"     {len(pt):>7} rows, {pt[-1]:.0f}s   altitude min {pa.min():.0f}  "
          f"max {pa.max():.0f} ft")
    print(f"data {args.data}")
    print(f"     {len(dt):>7} rows, {dt[-1]:.0f}s   altitude min {da.min():.0f}  "
          f"max {da.max():.0f} ft (MSL)")

    res = fit(pt, pa, dt, da, args.scale_min, args.scale_max)

    print(f"\nbest fit   data_t = {res['scale']:.4f} * pfd_t {res['shift']:+.1f} s")
    print(f"           stretch : {res['scale']:.4f}x  "
          f"(PFD {pt[-1]:.0f}s -> {pt[-1]*res['scale']:.0f}s)")
    print(f"           shift   : {res['shift']:+.1f} s")
    print(f"           correlation r = {res['r']:.3f}"
          + ("   (weak — altitude shapes do not really match)" if res['r'] < 0.6
             else "   (decent match)"))

    out = Path(args.out) if args.out else \
        ROOT / "exports" / "pfd" / f"{Path(args.pfd).stem}_alt_fit.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(pt, pa, dt, da, res).write_html(str(out))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
