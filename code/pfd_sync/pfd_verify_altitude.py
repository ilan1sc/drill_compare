"""Independent verifier for a PFD<->telemetry time sync: cross-check the
heading-based fit (pfd_compare_csv5.py) against a SECOND, unrelated
channel -- PFD radar altitude (RA) vs telemetry height-over-terrain.

If the heading fit is right, RA and HOT should also line up once the same
(t_scale, t_shift) is applied -- and this script finds its OWN
independent (t_scale, t_shift) on RA/HOT alone, so agreement between the
two fits (found from completely different signals) is strong confirmation,
and disagreement flags a sortie whose "good-looking" heading match might
still be wrong (e.g. 008, whose heading fit degrades in the back half).

    telemetry HOT : FFS1_Ownship.ESL_LHD_Landing_detection.
                    ESL_LHD_Landing_detection.Point_1_HOT   (full rate, every row)
    PFD  RA       : `ra` column of the OCR'd PFD csv (radar altitude readout)

Units/scale differ (HOT is metres, RA reads whatever the gauge shows), so
this fits a linear amplitude (gain, offset) alongside the time (t_scale,
t_shift) -- like pfd_compare_csv3.py's amplitude fit, but here on the
actual RA/HOT values directly (both already real, non-wrapping numbers --
no angle handling needed).

    python3 code/pfd_sync/pfd_verify_altitude.py --pfd <pfd_csv> --data <telemetry_csv>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent.parent
HOT = ("FFS1_Ownship.ESL_LHD_Landing_detection."
       "ESL_LHD_Landing_detection.Point_1_HOT")
GRID_HZ = 4.0
DT = 1.0 / GRID_HZ


def load_pfd_ra(path):
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy(float)
    r = pd.to_numeric(df["ra"], errors="coerce")
    r = r.interpolate(limit_direction="both").to_numpy()
    return t, r


def load_data_hot(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_numeric(df["Recording Time"], errors="coerce") / 1000.0
    h = pd.to_numeric(df[HOT], errors="coerce")
    m = t.notna() & h.notna()
    return t[m].to_numpy(), h[m].to_numpy()


def _on_grid(t, v, grid):
    return np.interp(grid, t, v, left=np.nan, right=np.nan)


def _fit_ab(x, y):
    a, b = np.polyfit(x, y, 1)
    yp = a * x + b
    r = np.corrcoef(yp, y)[0, 1] if np.std(yp) > 0 else 0.0
    return float(a), float(b), float(r)


def fit(pt, pr, dt, dr, scale_min, scale_max, n_coarse=61):
    dg = np.arange(dt[0], dt[-1], DT)
    dgv = np.nan_to_num(_on_grid(dt, dr - np.nanmean(dr), dg))

    def eval_scale(s):
        wt = pt * s
        pg = np.arange(wt[0], wt[-1], DT)
        pgv = np.nan_to_num(_on_grid(wt, pr - np.nanmean(pr), pg))
        c = signal.correlate(dgv, pgv, mode="full")
        lags = signal.correlation_lags(len(dgv), len(pgv), mode="full")
        shifts = dg[0] - pg[0] + lags * DT
        k = int(np.argmax(c))
        sh = float(shifts[k])
        # score at this (scale, shift): resample warped pfd onto data grid, fit a,b,r
        yy = _on_grid(pt * s + sh, pr, dg)
        ok = ~np.isnan(yy) & ~np.isnan(dr[0] * 0 + _on_grid(dt, dr, dg))
        dgy = _on_grid(dt, dr, dg)
        ok = ~np.isnan(yy) & ~np.isnan(dgy)
        if ok.sum() < 50:
            return (-1.0, s, sh, 1.0, 0.0, int(ok.sum()))
        a, b, r = _fit_ab(yy[ok], dgy[ok])
        return (r, s, sh, a, b, int(ok.sum()))

    coarse = [eval_scale(s) for s in np.linspace(scale_min, scale_max, n_coarse)]
    best = max(coarse, key=lambda x: x[0])
    step = (scale_max - scale_min) / (n_coarse - 1)
    fine = [eval_scale(s) for s in np.linspace(best[1] - step, best[1] + step, 41)]
    best = max([best] + fine, key=lambda x: x[0])
    r, s, sh, a, b, n = best
    return {"t_scale": s, "t_shift": sh, "amp_gain": a, "amp_offset": b, "r": r, "n": n}


def render(pt, pr, dt, dr, res, label, out_png):
    wt = pt * res["t_scale"] + res["t_shift"]
    wr = res["amp_gain"] * pr + res["amp_offset"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    axes[0].plot(dt, dr, color="#1f77b4", lw=0.8, label="telemetry HOT")
    ax0b = axes[0].twinx()
    ax0b.plot(pt, pr, color="#d62728", lw=0.8, label="PFD RA (raw units)")
    axes[0].set_title("RA vs HOT — as recorded, no alignment (separate y-scales)")

    axes[1].plot(dt, dr, color="#1f77b4", lw=0.9, label="telemetry HOT")
    axes[1].plot(wt, wr, color="#d62728", lw=0.9, label="PFD RA (aligned + rescaled)")
    axes[1].set_title(
        f"after   t: {res['t_scale']:.4f}·pfd {res['t_shift']:+.1f}s   "
        f"amp: {res['amp_gain']:.3f}·RA {res['amp_offset']:+.1f}   (r = {res['r']:.3f}, n={res['n']})")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("HOT (m)")
    ax0b.set_ylabel("RA (raw)")
    axes[1].set_ylabel("HOT (m) / RA rescaled to match")
    axes[1].set_xlabel("telemetry time (s)")
    fig.suptitle(f"PFD RA vs telemetry HOT — independent verifier{(' — ' + label) if label else ''}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def fit_at(pt, pr, dt, dr, t_scale, t_shift):
    """Don't search -- just fit amplitude (gain, offset) at a GIVEN
    (t_scale, t_shift), e.g. one already found from heading. RA/HOT is a
    poor channel to search on from scratch (it repeats a lot near the
    ground during pattern work, so cross-correlation locks onto false
    periodic peaks), but it's a fine channel to VERIFY an existing fit on."""
    dg = np.arange(dt[0], dt[-1], DT)
    yy = _on_grid(pt * t_scale + t_shift, pr, dg)
    dgy = _on_grid(dt, dr, dg)
    ok = ~np.isnan(yy) & ~np.isnan(dgy)
    a, b, r = _fit_ab(yy[ok], dgy[ok])
    return {"t_scale": t_scale, "t_shift": t_shift, "amp_gain": a,
            "amp_offset": b, "r": r, "n": int(ok.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.20)
    ap.add_argument("--t-scale", type=float, default=None,
                    help="verify at this t_scale (e.g. from pfd_compare_csv5.py) instead of searching")
    ap.add_argument("--t-shift", type=float, default=None,
                    help="verify at this t_shift instead of searching")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pt, pr = load_pfd_ra(args.pfd)
    dt, dr = load_data_hot(args.data)
    print(f"PFD RA  {len(pt)} rows, {pt[-1]:.0f}s, {np.nanmin(pr):.0f}..{np.nanmax(pr):.0f}")
    print(f"HOT     {len(dt)} rows, {dt[-1]:.0f}s, {np.nanmin(dr):.1f}..{np.nanmax(dr):.1f} m")

    if args.t_scale is not None and args.t_shift is not None:
        res = fit_at(pt, pr, dt, dr, args.t_scale, args.t_shift)
        print(f"VERIFY at heading fit: t_scale={res['t_scale']:.4f}  t_shift={res['t_shift']:+.1f}s"
              f"  r={res['r']:.3f}  n={res['n']}")
    else:
        res = fit(pt, pr, dt, dr, args.scale_min, args.scale_max)
        print(f"altitude-channel OWN fit: t_scale={res['t_scale']:.4f}  t_shift={res['t_shift']:+.1f}s"
              f"  r={res['r']:.3f}  n={res['n']}")
    render(pt, pr, dt, dr, res, args.label, args.out)
    print(f"Wrote {args.out}")
    return res


if __name__ == "__main__":
    main()
