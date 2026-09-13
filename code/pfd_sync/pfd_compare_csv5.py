"""Compare PFD heading with data heading and find BOTH the time offset and
the clock stretch -- on the actual heading (like csv4.py), not the heading
rate (like csv3.py).

csv3.py searches offset+stretch but on heading RATE (deg/s): the
derivative amplifies OCR jitter, so its own correlation number (r) can
read "weak" even when the two headings visually line up almost exactly
(confirmed 2026-09-12 on real check-ride data -- 006 scored r=0.44 "weak"
on rate but matches close to perfectly on heading itself).

csv4.py searches on heading directly with a wrap-safe score (mean cos of
the angular difference; 1.0 = identical) -- the right metric -- but only
a plain shift, no stretch. Over a ~45 min clip a 0.1% clock-rate
difference between the video encoder and the sim is ~2.7s of drift end to
end, enough to matter.

This is the two combined: csv4's direct, wrap-safe heading score, csv3's
two-parameter (t_scale, t_shift) search.

Model searched:

        pfd_t_in_data_clock  =  t_scale * pfd_t + t_shift

Search: for each candidate t_scale on a coarse grid, the best t_shift is
found in one shot via cross-correlation of the (cos, sin) heading
components (exact for the mean-cos score, and fast -- no per-shift loop);
the best-scoring scale is then refined on a finer grid around it.

    python3 code/pfd_sync/pfd_compare_csv5.py --pfd <pfd_csv> --data <telemetry_csv>
    python3 code/pfd_sync/pfd_compare_csv5.py --scale-min 0.95 --scale-max 1.05
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent.parent
DEF_PFD = ROOT / "exports" / "pfd" / "s1 c7 pfd rothman roitkof_pfd.csv"
DEF_DATA = ("/home/ilan/Documents/flight school/data/csv_data/1_26/"
            "CSV_3.6.26_check1/007/Elements_0.csv")
HDG = ("FFS1_Ownship.LHD_ESL_Flight_Model_Data."
       "LHD_ESL_Flight_Model_Data.Heading")
GRID_HZ = 4.0
DT = 1.0 / GRID_HZ


def load_pfd(path):
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy(float)
    h = pd.to_numeric(df["heading"], errors="coerce").clip(0, 360)
    h = h.interpolate(limit_direction="both").to_numpy()
    return t, h


def load_data(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_numeric(df["Recording Time"], errors="coerce") / 1000.0
    h = np.degrees(pd.to_numeric(df[HDG], errors="coerce")) % 360.0   # rad -> deg
    m = t.notna() & pd.Series(h).notna()
    return t[m].to_numpy(), np.asarray(h)[m.to_numpy()]


def _cos_sin_on_grid(t, h, grid):
    """Resample heading onto `grid` through the unit circle (wrap-safe)."""
    c = np.interp(grid, t, np.cos(np.radians(h)), left=np.nan, right=np.nan)
    s = np.interp(grid, t, np.sin(np.radians(h)), left=np.nan, right=np.nan)
    return c, s


def _best_shift_for_scale(pt, ph, dg, dc, ds, scale):
    """Best t_shift (in data-clock seconds) for a given t_scale, found by
    cross-correlating the cos/sin channels -- exact argmax of the mean-cos
    score over the overlap, and O(n log n) instead of a shift loop."""
    wt = pt * scale
    pg = np.arange(wt[0], wt[-1], DT)
    pc, ps = _cos_sin_on_grid(wt, ph, pg)
    pc = np.nan_to_num(pc)
    ps = np.nan_to_num(ps)
    cc = signal.correlate(dc, pc, mode="full")
    cs = signal.correlate(ds, ps, mode="full")
    c = cc + cs
    lags = signal.correlation_lags(len(dc), len(pc), mode="full")
    shifts = dg[0] - pg[0] + lags * DT
    k = int(np.argmax(c))
    return float(shifts[k])


def _score(pt, ph, dt, dh, scale, shift):
    lo = max(pt[0] * scale + shift, dt[0])
    hi = min(pt[-1] * scale + shift, dt[-1])
    if hi <= lo:
        return -2.0, 0
    grid = np.arange(lo, hi, DT)
    dg = _grid_deg(dt, dh, grid)
    pg = _grid_deg(pt * scale + shift, ph, grid)
    diff = np.radians(pg - dg)
    return float(np.mean(np.cos(diff))), len(grid)


def _grid_deg(t, h, grid):
    c = np.interp(grid, t, np.cos(np.radians(h)))
    s = np.interp(grid, t, np.sin(np.radians(h)))
    return np.degrees(np.arctan2(s, c)) % 360.0


def fit(pt, ph, dt, dh, scale_min, scale_max, n_coarse=61):
    dg = np.arange(dt[0], dt[-1], DT)
    dc, ds = _cos_sin_on_grid(dt, dh, dg)
    dc = np.nan_to_num(dc)
    ds = np.nan_to_num(ds)

    def eval_scale(s):
        sh = _best_shift_for_scale(pt, ph, dg, dc, ds, s)
        score, n = _score(pt, ph, dt, dh, s, sh)
        return (score, s, sh, n)

    coarse = [eval_scale(s) for s in np.linspace(scale_min, scale_max, n_coarse)]
    best = max(coarse, key=lambda r: r[0])
    step = (scale_max - scale_min) / (n_coarse - 1)
    fine = [eval_scale(s) for s in np.linspace(best[1] - step, best[1] + step, 41)]
    best = max([best] + fine, key=lambda r: r[0])

    score, scale, shift, n = best
    rms_deg = float(np.degrees(np.arccos(np.clip(score, -1, 1))))
    return {"t_scale": scale, "t_shift": shift, "score": score,
            "rms_deg": rms_deg, "n": n}


def _wrapped_for_plot(t, h):
    """insert NaN at 0<->360 jumps so the line doesn't streak vertically."""
    h = np.asarray(h, float)
    jump = np.abs(np.diff(h)) > 180
    ti, hi = [t[0]], [h[0]]
    for k in range(1, len(h)):
        if jump[k - 1]:
            ti.append(t[k]); hi.append(np.nan)
        ti.append(t[k]); hi.append(h[k])
    return np.array(ti), np.array(hi)


def build_figure(pt, ph, dt, dh, res, label=""):
    wt = pt * res["t_scale"] + res["t_shift"]
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.12, shared_xaxes=False,
        subplot_titles=[
            "heading (deg) — as recorded, no alignment",
            f"after   t: {res['t_scale']:.4f}·pfd {res['t_shift']:+.1f}s   "
            f"(mean-cos {res['score']:.4f}, ~{res['rms_deg']:.1f}° error, n={res['n']})"])
    dtx, dhx = _wrapped_for_plot(dt, dh)
    for row, ptw in ((1, pt), (2, wt)):
        ptx, phx = _wrapped_for_plot(ptw, ph)
        fig.add_trace(go.Scatter(x=dtx, y=dhx, name="data",
                                 line=dict(color="#1f77b4", width=1),
                                 showlegend=(row == 1)), row=row, col=1)
        fig.add_trace(go.Scatter(x=ptx, y=phx, name="PFD",
                                 line=dict(color="#d62728", width=1),
                                 showlegend=(row == 1)), row=row, col=1)
    fig.update_layout(height=680, width=1400, template="plotly_white",
                      legend=dict(orientation="h"),
                      title=f"PFD vs data heading — offset + stretch fit{(' — ' + label) if label else ''}")
    fig.update_yaxes(title_text="heading (deg)", range=[0, 360])
    fig.update_xaxes(title_text="data time (s)")
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd", default=str(DEF_PFD))
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.20)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="output HTML path")
    ap.add_argument("--no-aligned-csv", action="store_true",
                    help="skip writing the data CSV re-timed to the PFD video clock")
    args = ap.parse_args()

    pt, ph = load_pfd(Path(args.pfd))
    dt, dh = load_data(args.data)
    print(f"PFD  {args.pfd}\n     {len(pt)} rows, {pt[-1]:.0f}s, heading {ph.min():.0f}..{ph.max():.0f}")
    print(f"data {args.data}\n     {len(dt)} rows, {dt[-1]:.0f}s, heading {dh.min():.0f}..{dh.max():.0f}")

    res = fit(pt, ph, dt, dh, args.scale_min, args.scale_max)
    print(f"\nbest fit")
    print(f"  time  data_t = {res['t_scale']:.4f} * pfd_t {res['t_shift']:+.1f} s"
          f"   (stretch {res['t_scale']:.4f}x, move {res['t_shift']:+.1f}s)")
    print(f"  match : mean-cos {res['score']:.4f}  (~{res['rms_deg']:.1f} deg mean error, "
          f"n={res['n']} samples)"
          + ("   strong" if res["score"] > 0.9 else
             "   weak — headings do not really match"))

    out = Path(args.out) if args.out else \
        ROOT / "exports" / "pfd" / f"{Path(args.pfd).stem}_hdg_fit5.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(pt, ph, dt, dh, res, args.label).write_html(str(out))
    print(f"\nWrote {out}")

    if not args.no_aligned_csv:
        raw = pd.read_csv(args.data, sep=None, engine="python")
        data_s = pd.to_numeric(raw[raw.columns[0]], errors="coerce") / 1000.0
        pfd_s = (data_s - res["t_shift"]) / res["t_scale"]
        aligned = raw.copy()
        aligned.insert(0, "PFD Time (s)", pfd_s.round(4))
        aligned.insert(1, "PFD Time (ms)", (pfd_s * 1000.0).round(1))
        acsv = out.with_name(f"{Path(args.data).stem}_in_{Path(args.pfd).stem}_time.csv")
        aligned.to_csv(acsv, index=False)
        print(f"Wrote {acsv}")
        print(f"  data CSV re-timed to the PFD video clock  "
              f"(PFD Time = (Recording Time/1000 {-res['t_shift']:+.2f}) / {res['t_scale']:.4f})")


if __name__ == "__main__":
    main()
