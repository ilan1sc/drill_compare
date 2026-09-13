"""Compare PFD heading with data heading and find the small time offset.

Minimal version: heading only, a plain time shift (no stretch, no
amplitude rescale). The offset between the PFD video CSV and the data CSV
is normally just a few seconds, so the search is a narrow window around
zero.

    PFD  heading : `heading` column, already in degrees.
    data heading : `...LHD_ESL_Flight_Model_Data.Heading`, radians -> * 180/pi.

Both are compared on a common time grid with a wrap-aware score
(mean cos of the angular difference; 1.0 = identical). The reported
offset is the seconds to ADD to PFD time so it lines up with the data
CSV. Before/after graphs are written to HTML.

Usage
-----
    python3 code/pfd_sync/pfd_compare_csv4.py
    python3 code/pfd_sync/pfd_compare_csv4.py --pfd "exports/pfd/s1 c8 pdf sorek lifman_pfd.csv" \\
        --data ".../CSV_3.6.26_check1/008/Elements_0.csv"
    python3 code/pfd_sync/pfd_compare_csv4.py --max-shift 120
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent.parent
DEF_PFD = ROOT / "exports" / "pfd" / "s1 c7 pfd rothman roitkof_pfd.csv"
DEF_DATA = ("/home/ilan/Documents/flight school/data/csv_data/1_26/"
            "CSV_3.6.26_check1/003/Elements_0.csv")
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


def _grid(t, h, grid):
    """Resample heading onto `grid` through the unit circle (wrap-safe)."""
    c = np.interp(grid, t, np.cos(np.radians(h)))
    s = np.interp(grid, t, np.sin(np.radians(h)))
    return np.degrees(np.arctan2(s, c)) % 360.0


def find_offset(pt, ph, dt, dh, max_shift, step):
    lo = max(pt[0], dt[0])
    hi = min(pt[-1], dt[-1])
    grid = np.arange(lo, hi, DT)
    dg = _grid(dt, dh, grid)
    shifts = np.arange(-max_shift, max_shift + step, step)
    best = (-2.0, 0.0)
    curve = []
    for sh in shifts:
        pg = _grid(pt + sh, ph, grid)
        score = float(np.mean(np.cos(np.radians(pg - dg))))
        curve.append((sh, score))
        if score > best[0]:
            best = (score, float(sh))
    score, off = best
    rms_deg = float(np.degrees(np.arccos(np.clip(score, -1, 1))))
    return {"offset": off, "score": score, "rms_deg": rms_deg,
            "curve": np.array(curve)}


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


def build_figure(pt, ph, dt, dh, res):
    off = res["offset"]
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.13, shared_xaxes=False,
        subplot_titles=[
            "heading (deg) — as recorded",
            f"after  PFD time {off:+.2f} s   "
            f"(mean-cos {res['score']:.4f}, ~{res['rms_deg']:.1f}° error)"])
    dtx, dhx = _wrapped_for_plot(dt, dh)
    for row, shift in ((1, 0.0), (2, off)):
        ptx, phx = _wrapped_for_plot(pt + shift, ph)
        fig.add_trace(go.Scatter(x=dtx, y=dhx, name="data",
                                 line=dict(color="#1f77b4", width=1),
                                 showlegend=(row == 1)), row=row, col=1)
        fig.add_trace(go.Scatter(x=ptx, y=phx, name="PFD",
                                 line=dict(color="#d62728", width=1),
                                 showlegend=(row == 1)), row=row, col=1)
    fig.update_layout(height=680, width=1400, template="plotly_white",
                      legend=dict(orientation="h"),
                      title="PFD vs data heading — time-offset fit")
    fig.update_yaxes(title_text="heading (deg)", range=[0, 360])
    fig.update_xaxes(title_text="data time (s)")
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd", default=str(DEF_PFD))
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--max-shift", type=float, default=60.0,
                    help="search PFD offset within +/- this many seconds (default 60)")
    ap.add_argument("--step", type=float, default=0.25, help="offset step, s (default 0.25)")
    ap.add_argument("--out", help="output HTML path")
    ap.add_argument("--no-aligned-csv", action="store_true",
                    help="skip writing the data CSV re-timed to the PFD video clock")
    args = ap.parse_args()

    pt, ph = load_pfd(Path(args.pfd))
    dt, dh = load_data(args.data)
    print(f"PFD  {args.pfd}\n     {len(pt)} rows, {pt[-1]:.0f}s, heading {ph.min():.0f}..{ph.max():.0f}")
    print(f"data {args.data}\n     {len(dt)} rows, {dt[-1]:.0f}s, heading {dh.min():.0f}..{dh.max():.0f}")

    res = find_offset(pt, ph, dt, dh, args.max_shift, args.step)
    print(f"\nbest offset : {res['offset']:+.2f} s  (add to PFD time)")
    print(f"match       : mean-cos {res['score']:.4f}  (~{res['rms_deg']:.1f} deg mean error)"
          + ("   strong" if res['score'] > 0.9 else
             "   weak — headings do not match in this window"))

    out = Path(args.out) if args.out else \
        ROOT / "exports" / "pfd" / f"{Path(args.pfd).stem}_hdg_offset.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(pt, ph, dt, dh, res).write_html(str(out))
    print(f"\nWrote {out}")

    if not args.no_aligned_csv:
        raw = pd.read_csv(args.data, sep=None, engine="python")
        data_s = pd.to_numeric(raw[raw.columns[0]], errors="coerce") / 1000.0
        pfd_s = data_s - res["offset"]        # data_t ~= pfd_t + offset  ->  pfd_t = data_t - offset
        aligned = raw.copy()
        aligned.insert(0, "PFD Time (s)", pfd_s.round(4))
        aligned.insert(1, "PFD Time (ms)", (pfd_s * 1000.0).round(1))
        acsv = out.with_name(f"{Path(args.data).stem}_in_{Path(args.pfd).stem}_time.csv")
        aligned.to_csv(acsv, index=False)
        print(f"Wrote {acsv}")
        print(f"  data CSV re-timed to the PFD video clock  "
              f"(PFD Time = Recording Time/1000 {-res['offset']:+.2f} s)")


if __name__ == "__main__":
    main()
