#!/usr/bin/env python3
"""Plot the actual PFD-vs-telemetry HEADING (degrees), not the heading-rate
used internally to find the fit.

pfd_compare_csv3.py's own plot shows deg/s (what the correlation searches
on); this shows the heading angle itself, before and after applying the
time alignment it found -- the intuitive "do these two headings actually
match" view.

    python3 code/pfd_sync/pfd_heading_compare_plot.py --pfd <pfd_csv> --data <telemetry_csv>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfd_compare_csv3 import load_pfd_heading, load_data_heading, fit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent


def _wrap_break(t, h):
    """Insert a NaN wherever heading wraps 360<->0, so plotly doesn't draw
    a vertical line across the jump."""
    h = np.asarray(h, float)
    jumps = np.abs(np.diff(h)) > 180
    t_out, h_out = [t[0]], [h[0]]
    for i in range(1, len(h)):
        if jumps[i - 1]:
            t_out.append(t[i])
            h_out.append(np.nan)
        t_out.append(t[i])
        h_out.append(h[i])
    return np.asarray(t_out), np.asarray(h_out)


def build_figure(pt, ph, dt, dh, res, label=""):
    wt = pt * res["t_scale"] + res["t_shift"]

    pt_b, ph_b = _wrap_break(pt, ph)
    dt_b, dh_b = _wrap_break(dt, dh)
    wt_b, wh_b = _wrap_break(wt, ph)

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.1,
        subplot_titles=[
            "heading (deg) — as recorded, no time alignment",
            f"heading (deg) — after time align   "
            f"t: {res['t_scale']:.4f}·pfd {res['t_shift']:+.1f}s   (r = {res['r']:.3f}, on rate)"])

    fig.add_trace(go.Scattergl(x=dt_b, y=dh_b, name="telemetry", mode="lines",
                               line=dict(color="#1f77b4", width=1)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=pt_b, y=ph_b, name="PFD (OCR)", mode="lines",
                               line=dict(color="#d62728", width=1)), row=1, col=1)

    fig.add_trace(go.Scattergl(x=dt_b, y=dh_b, name="telemetry", showlegend=False,
                               mode="lines", line=dict(color="#1f77b4", width=1)), row=2, col=1)
    fig.add_trace(go.Scattergl(x=wt_b, y=wh_b, name="PFD (OCR, aligned)", showlegend=False,
                               mode="lines", line=dict(color="#d62728", width=1)), row=2, col=1)

    fig.update_layout(height=760, width=1500, template="plotly_white",
                      legend=dict(orientation="h"),
                      title=f"PFD vs telemetry HEADING{(' — ' + label) if label else ''}")
    fig.update_xaxes(title_text="telemetry time (s)")
    fig.update_yaxes(title_text="heading (deg)", range=[0, 360])
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.20)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="output HTML path")
    args = ap.parse_args()

    pt, ph = load_pfd_heading(Path(args.pfd))
    dt, dh = load_data_heading(args.data)
    res = fit(pt, ph, dt, dh, args.scale_min, args.scale_max)
    print(f"t_scale={res['t_scale']:.4f}  t_shift={res['t_shift']:+.1f}s  r={res['r']:.3f}")

    out = Path(args.out) if args.out else \
        ROOT / "exports" / "pfd" / f"{Path(args.pfd).stem}_hdg_compare.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(pt, ph, dt, dh, res, args.label).write_html(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
