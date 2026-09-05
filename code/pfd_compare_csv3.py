"""Fit the PFD video CSV to the data CSV on HEADING, allowing the video
timeline to be moved / stretched and the amplitude to be rescaled.

Like pfd_compare_csv2.py but on heading instead of altitude. PFD values
are kept in their own units (degrees, as the ICD/CSV gives them); the
data heading is converted from radians only so the two can be compared.

Heading wraps, so the matching is done on HEADING RATE (deg/s of the
unwrapped, wrap-safe-smoothed heading): turns show up as peaks, straight
flight sits near zero, and there is no 359->0 discontinuity.

Model searched:

        data_time      ~=  t_scale * pfd_time + t_shift
        data_headrate  ~=  a * pfd_headrate + b

t_scale / t_shift are found by cross-correlation over a grid of t_scale;
a (amplitude gain) and b (amplitude offset) are the least-squares fit of
the two rate traces at the best alignment.

Usage
-----
    python3 code/pfd_compare_csv3.py
    python3 code/pfd_compare_csv3.py --pfd "exports/pfd/s1 c8 pdf sorek lifman_pfd.csv" \\
        --data ".../CSV_3.6.26_check1/008/Elements_0.csv"
    python3 code/pfd_compare_csv3.py --scale-min 0.95 --scale-max 1.05

    cd "/home/ilan/Documents/flight school/drill_compare" && python3 code/pfd_compare_csv3.py
python3 code/pfd_compare_csv3.py \
  --pfd "exports/pfd/s1 c8 pdf sorek lifman_pfd.csv" \
  --data "/home/ilan/Documents/flight school/data/csv_data/1_26/CSV_3.6.26_check1/008/Elements_0.csv"

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
DT = 0.5                       # s, grid the rate traces are compared on
SMOOTH_S = 3.0                 # s, wrap-safe heading smoothing window


def _circ_smooth(deg, win):
    ang = np.radians(np.asarray(deg, float))
    c = pd.Series(np.cos(ang)).rolling(win, center=True, min_periods=1).mean()
    s = pd.Series(np.sin(ang)).rolling(win, center=True, min_periods=1).mean()
    return np.degrees(np.arctan2(s.to_numpy(), c.to_numpy()))


def _rate_on_grid(t, deg, grid, smooth_win):
    """wrap-safe smooth -> unwrap -> resample -> deg/s."""
    sm = _circ_smooth(deg, smooth_win)
    unw = np.degrees(np.unwrap(np.radians(sm)))
    y = np.interp(grid, t, unw)
    return np.gradient(y, DT)


def load_pfd_heading(path):
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy(float)
    h = pd.to_numeric(df["heading"], errors="coerce").clip(0, 360)
    h = _circ_smooth(h.interpolate(limit_direction="both"), 9)  # kill OCR jitter
    return t, h % 360.0


def load_data_heading(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_numeric(df["Recording Time"], errors="coerce") / 1000.0
    h = np.degrees(pd.to_numeric(df[FM + "Heading"], errors="coerce"))
    m = t.notna() & pd.Series(h).notna()
    return t[m].to_numpy(), (np.asarray(h)[m.to_numpy()] % 360.0)


def _fit_ab(x, y):
    """least squares y ~ a*x + b, plus pearson r."""
    a, b = np.polyfit(x, y, 1)
    yp = a * x + b
    r = np.corrcoef(yp, y)[0, 1] if np.std(yp) > 0 else 0.0
    return float(a), float(b), float(r)


def _best_shift(pt, pr_t, pr, gd, dr):
    gp = np.arange(pr_t[0], pr_t[-1], DT)
    yp = np.interp(gp, pr_t, pr)
    c = signal.correlate(dr - dr.mean(), yp - yp.mean(), mode="full")
    lags = signal.correlation_lags(len(dr), len(yp), mode="full")
    shifts = gd[0] - gp[0] + lags * DT
    k = int(np.argmax(c))
    return float(shifts[k])


def fit(pt, ph, dt, dh, scale_min, scale_max, n_coarse=81):
    gd = np.arange(dt[0], dt[-1], DT)
    dr = _rate_on_grid(dt, dh, gd, max(1, round(SMOOTH_S / DT)))

    def score_scale(s):
        wt = pt * s
        pr = _rate_on_grid(wt, ph, np.arange(wt[0], wt[-1], DT),
                           max(1, round(SMOOTH_S / DT)))
        pr_t = np.arange(wt[0], wt[-1], DT)
        sh = _best_shift(wt, pr_t, pr, gd, dr)
        # resample warped+shifted pfd rate onto data grid; fit a,b + r
        yy = np.interp(gd, pr_t + sh, pr, left=np.nan, right=np.nan)
        ok = ~np.isnan(yy)
        if ok.sum() < 50:
            return (-1.0, s, sh, 1.0, 0.0)
        a, b, r = _fit_ab(yy[ok], dr[ok])
        return (r, s, sh, a, b)

    best = max(score_scale(s) for s in np.linspace(scale_min, scale_max, n_coarse))
    step = (scale_max - scale_min) / (n_coarse - 1)
    best = max([best] + [score_scale(s)
                         for s in np.linspace(best[1] - step, best[1] + step, 41)])
    r, s, sh, a, b = best
    return {"t_scale": s, "t_shift": sh, "amp_gain": a, "amp_offset": b, "r": r}


def build_figure(pt, ph, dt, dh, res):
    gd = np.arange(dt[0], dt[-1], DT)
    win = max(1, round(SMOOTH_S / DT))
    dr = _rate_on_grid(dt, dh, gd, win)
    wt = pt * res["t_scale"] + res["t_shift"]
    pr_raw = _rate_on_grid(pt, ph, np.arange(pt[0], pt[-1], DT), win)
    pr_raw_t = np.arange(pt[0], pt[-1], DT)
    pr_w = _rate_on_grid(wt, ph, np.arange(wt[0], wt[-1], DT), win)
    pr_w_t = np.arange(wt[0], wt[-1], DT)
    pr_w_amp = res["amp_gain"] * pr_w + res["amp_offset"]

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.12,
        subplot_titles=[
            "heading rate (deg/s) — as recorded",
            f"after   t: {res['t_scale']:.4f}·pfd {res['t_shift']:+.1f}s   "
            f"amp: {res['amp_gain']:.3f}·pfd {res['amp_offset']:+.3f}   "
            f"(r = {res['r']:.3f})"])
    fig.add_trace(go.Scatter(x=gd, y=dr, name="data", line=dict(color="#1f77b4", width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=pr_raw_t, y=pr_raw, name="PFD",
                             line=dict(color="#d62728", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gd, y=dr, name="data", showlegend=False,
                             line=dict(color="#1f77b4", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=pr_w_t, y=pr_w_amp, name="PFD fitted", showlegend=False,
                             line=dict(color="#d62728", width=1)), row=2, col=1)
    fig.update_layout(height=700, width=1400, template="plotly_white",
                      legend=dict(orientation="h"),
                      title="PFD vs data heading — move / stretch / rescale fit")
    fig.update_xaxes(title_text="data time (s)")
    fig.update_yaxes(title_text="deg/s")
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd", default=str(DEF_PFD))
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.20)
    ap.add_argument("--out", help="output HTML path")
    ap.add_argument("--no-aligned-csv", action="store_true",
                    help="skip writing the data CSV re-timed to the PFD video clock")
    args = ap.parse_args()

    pt, ph = load_pfd_heading(Path(args.pfd))
    dt, dh = load_data_heading(args.data)

    print(f"PFD  {args.pfd}")
    print(f"     {len(pt):>7} rows, {pt[-1]:.0f}s   heading {ph.min():.0f}..{ph.max():.0f} deg")
    print(f"data {args.data}")
    print(f"     {len(dt):>7} rows, {dt[-1]:.0f}s   heading {dh.min():.0f}..{dh.max():.0f} deg")

    res = fit(pt, ph, dt, dh, args.scale_min, args.scale_max)

    print(f"\nbest fit")
    print(f"  time      data_t = {res['t_scale']:.4f} * pfd_t {res['t_shift']:+.1f} s"
          f"   (stretch {res['t_scale']:.4f}x, move {res['t_shift']:+.1f}s)")
    print(f"  amplitude data_headrate = {res['amp_gain']:.3f} * pfd_headrate "
          f"{res['amp_offset']:+.3f}")
    print(f"  correlation r = {res['r']:.3f}"
          + ("   (strong match)" if res["r"] >= 0.7 else
             "   (weak — headings do not really match)"))

    out = Path(args.out) if args.out else \
        ROOT / "exports" / "pfd" / f"{Path(args.pfd).stem}_hdg_fit.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(pt, ph, dt, dh, res).write_html(str(out))
    print(f"\nWrote {out}")

    if not args.no_aligned_csv:
        raw = pd.read_csv(args.data, sep=None, engine="python")
        t_ms = pd.to_numeric(raw[raw.columns[0]], errors="coerce")
        # invert  data_t = t_scale*pfd_t + t_shift   ->   pfd_t = (data_t - shift)/scale
        pfd_s = (t_ms / 1000.0 - res["t_shift"]) / res["t_scale"]
        aligned = raw.copy()
        aligned.insert(0, "PFD Time (s)", pfd_s.round(4))
        aligned.insert(1, "PFD Time (ms)", (pfd_s * 1000.0).round(1))
        acsv = out.with_name(f"{Path(args.data).stem}_in_{Path(args.pfd).stem}_time.csv")
        aligned.to_csv(acsv, index=False)
        print(f"Wrote {acsv}")
        print(f"  data CSV with two leading columns in PFD-video time "
              f"(pfd_t = (data_t {-res['t_shift']:+.1f}) / {res['t_scale']:.4f})")


if __name__ == "__main__":
    main()
