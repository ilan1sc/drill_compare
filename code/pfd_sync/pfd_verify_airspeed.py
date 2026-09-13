"""Second independent verifier: PFD airspeed vs telemetry TrueAirSpeed.

Unlike RA/HOT (pfd_verify_altitude.py), this is the same physical
quantity on both sides (just kt vs m/s), sampled at the same ~20 Hz as
heading -- a genuinely independent second check, and unlike RA/HOT it's
not confined near a repeating low value, so (unlike RA/HOT) it's usable
for its OWN from-scratch (t_scale, t_shift) search too, not just
verifying a fit found elsewhere.

    telemetry TAS : FFS1_Ownship.LHD_ESL_Flight_Model_Data.
                    LHD_ESL_Flight_Model_Data.TrueAirSpeed   (m/s -> kt)
    PFD  airspeed : `airspeed` column of the OCR'd PFD csv (kt, as shown)

    python3 code/pfd_sync/pfd_verify_airspeed.py --pfd <pfd_csv> --data <telemetry_csv>
    python3 code/pfd_sync/pfd_verify_airspeed.py --pfd <pfd_csv> --data <telemetry_csv> \\
        --t-scale 0.9990 --t-shift 0.5      # verify at a fit found elsewhere
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
TAS = ("FFS1_Ownship.LHD_ESL_Flight_Model_Data."
       "LHD_ESL_Flight_Model_Data.TrueAirSpeed")
MPS_TO_KT = 1.943844
GRID_HZ = 4.0
DT = 1.0 / GRID_HZ


def load_pfd_ias(path, lo=60, hi=120):
    """The airspeed readout is a rolling drum -- OCR either reads it clean
    or drops to 0 / a misread stacked-digit value (e.g. 149, 169, 200)
    mid-roll. Keep only samples in the plausible [lo, hi] kt band and
    interpolate through everything else (0 included) from the nearest
    valid neighbours either side."""
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy(float)
    v = pd.to_numeric(df["airspeed"], errors="coerce")
    n_raw = v.notna().sum()
    v = v.where((v >= lo) & (v <= hi))
    n_kept = v.notna().sum()
    print(f"  airspeed: kept {n_kept}/{n_raw} raw OCR reads within [{lo},{hi}] kt, "
          f"interpolated the rest")
    v = v.interpolate(limit_direction="both").to_numpy()
    return t, v


def load_data_tas(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_numeric(df["Recording Time"], errors="coerce") / 1000.0
    v = pd.to_numeric(df[TAS], errors="coerce") * MPS_TO_KT
    m = t.notna() & v.notna()
    return t[m].to_numpy(), v[m].to_numpy()


def _on_grid(t, v, grid):
    return np.interp(grid, t, v, left=np.nan, right=np.nan)


def _fit_ab(x, y):
    a, b = np.polyfit(x, y, 1)
    yp = a * x + b
    r = np.corrcoef(yp, y)[0, 1] if np.std(yp) > 0 else 0.0
    return float(a), float(b), float(r)


def _score_at(pt, pv, dt, dv, scale, shift):
    dg = np.arange(dt[0], dt[-1], DT)
    yy = _on_grid(pt * scale + shift, pv, dg)
    dgy = _on_grid(dt, dv, dg)
    ok = ~np.isnan(yy) & ~np.isnan(dgy)
    if ok.sum() < 50:
        return -1.0, 1.0, 0.0, int(ok.sum())
    a, b, r = _fit_ab(yy[ok], dgy[ok])
    return r, a, b, int(ok.sum())


def fit(pt, pv, dt, dv, scale_min, scale_max, n_coarse=61):
    dg = np.arange(dt[0], dt[-1], DT)
    dgv = np.nan_to_num(_on_grid(dt, dv - np.nanmean(dv), dg))

    def eval_scale(s):
        wt = pt * s
        pg = np.arange(wt[0], wt[-1], DT)
        pgv = np.nan_to_num(_on_grid(wt, pv - np.nanmean(pv), pg))
        c = signal.correlate(dgv, pgv, mode="full")
        lags = signal.correlation_lags(len(dgv), len(pgv), mode="full")
        shifts = dg[0] - pg[0] + lags * DT
        k = int(np.argmax(c))
        sh = float(shifts[k])
        r, a, b, n = _score_at(pt, pv, dt, dv, s, sh)
        return (r, s, sh, a, b, n)

    coarse = [eval_scale(s) for s in np.linspace(scale_min, scale_max, n_coarse)]
    best = max(coarse, key=lambda x: x[0])
    step = (scale_max - scale_min) / (n_coarse - 1)
    fine = [eval_scale(s) for s in np.linspace(best[1] - step, best[1] + step, 41)]
    best = max([best] + fine, key=lambda x: x[0])
    r, s, sh, a, b, n = best
    return {"t_scale": s, "t_shift": sh, "amp_gain": a, "amp_offset": b, "r": r, "n": n}


def fit_at(pt, pv, dt, dv, t_scale, t_shift):
    r, a, b, n = _score_at(pt, pv, dt, dv, t_scale, t_shift)
    return {"t_scale": t_scale, "t_shift": t_shift, "amp_gain": a,
            "amp_offset": b, "r": r, "n": n}


def render(pt, pv, dt, dv, res, label, out_png):
    wt = pt * res["t_scale"] + res["t_shift"]
    wv = res["amp_gain"] * pv + res["amp_offset"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharey=True)
    axes[0].plot(dt, dv, color="#1f77b4", lw=0.7, label="telemetry TAS (kt)")
    axes[0].plot(pt, pv, color="#d62728", lw=0.7, label="PFD airspeed (raw kt)")
    axes[0].set_title("airspeed (kt) — as recorded, no alignment")
    axes[0].legend(loc="upper right")

    axes[1].plot(dt, dv, color="#1f77b4", lw=0.8, label="telemetry TAS")
    axes[1].plot(wt, wv, color="#d62728", lw=0.8, label="PFD airspeed (aligned)")
    axes[1].set_title(
        f"after   t: {res['t_scale']:.4f}·pfd {res['t_shift']:+.1f}s   "
        f"amp: {res['amp_gain']:.3f}·kt {res['amp_offset']:+.1f}   (r = {res['r']:.3f}, n={res['n']})")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_ylabel("airspeed (kt)")
    axes[1].set_xlabel("telemetry time (s)")
    fig.suptitle(f"PFD airspeed vs telemetry TAS — verifier{(' — ' + label) if label else ''}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.20)
    ap.add_argument("--t-scale", type=float, default=None)
    ap.add_argument("--t-shift", type=float, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pt, pv = load_pfd_ias(args.pfd)
    dt, dv = load_data_tas(args.data)
    print(f"PFD airspeed  {len(pt)} rows, {pt[-1]:.0f}s, {np.nanmin(pv):.0f}..{np.nanmax(pv):.0f} kt")
    print(f"telemetry TAS {len(dt)} rows, {dt[-1]:.0f}s, {np.nanmin(dv):.0f}..{np.nanmax(dv):.0f} kt")

    if args.t_scale is not None and args.t_shift is not None:
        res = fit_at(pt, pv, dt, dv, args.t_scale, args.t_shift)
        print(f"VERIFY at given fit: t_scale={res['t_scale']:.4f}  t_shift={res['t_shift']:+.1f}s"
              f"  r={res['r']:.3f}  n={res['n']}")
    else:
        res = fit(pt, pv, dt, dv, args.scale_min, args.scale_max)
        print(f"airspeed-channel OWN fit: t_scale={res['t_scale']:.4f}  t_shift={res['t_shift']:+.1f}s"
              f"  r={res['r']:.3f}  n={res['n']}")
    render(pt, pv, dt, dv, res, args.label, args.out)
    print(f"Wrote {args.out}")
    return res


if __name__ == "__main__":
    main()
