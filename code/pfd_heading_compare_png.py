#!/usr/bin/env python3
"""Same plot as pfd_heading_compare_plot.py (actual heading in degrees,
before/after time-align) but rendered to a static PNG with matplotlib --
no browser/kaleido dependency, so it can be viewed inline.

    python3 code/pfd_heading_compare_png.py --pfd <pfd_csv> --data <telemetry_csv> --out <png>
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfd_compare_csv3 import load_pfd_heading, load_data_heading, fit  # noqa: E402


def _wrap_break(t, h):
    h = np.asarray(h, float)
    out_h = h.copy()
    jumps = np.abs(np.diff(h)) > 180
    out_h[1:][jumps] = np.nan
    return t, out_h


def render(pt, ph, dt, dh, res, label, out_png):
    wt = pt * res["t_scale"] + res["t_shift"]
    dt_p, dh_p = _wrap_break(dt, dh)
    pt_p, ph_p = _wrap_break(pt, ph)
    wt_p, wh_p = _wrap_break(wt, ph)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharey=True)
    axes[0].plot(dt_p, dh_p, color="#1f77b4", lw=0.8, label="telemetry")
    axes[0].plot(pt_p, ph_p, color="#d62728", lw=0.8, label="PFD (OCR)")
    axes[0].set_title("heading (deg) — as recorded, no time alignment")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(dt_p, dh_p, color="#1f77b4", lw=0.8, label="telemetry")
    axes[1].plot(wt_p, wh_p, color="#d62728", lw=0.8, label="PFD (OCR, aligned)")
    axes[1].set_title(
        f"heading (deg) — after time align   "
        f"t: {res['t_scale']:.4f}·pfd {res['t_shift']:+.1f}s   (r = {res['r']:.3f}, on rate)"
    )
    axes[1].legend(loc="upper right", ncol=2)

    for ax in axes:
        ax.set_ylim(0, 360)
        ax.set_ylabel("heading (deg)")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("telemetry time (s)")

    fig.suptitle(f"PFD vs telemetry HEADING{(' — ' + label) if label else ''}", y=0.995)
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
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pt, ph = load_pfd_heading(Path(args.pfd))
    dt, dh = load_data_heading(args.data)
    res = fit(pt, ph, dt, dh, args.scale_min, args.scale_max)
    print(f"t_scale={res['t_scale']:.4f}  t_shift={res['t_shift']:+.1f}s  r={res['r']:.3f}")
    render(pt, ph, dt, dh, res, args.label, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
