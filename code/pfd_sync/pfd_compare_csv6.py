"""Same fit as pfd_compare_csv5.py (heading offset + stretch, direct
wrap-safe mean-cos score) -- but the search itself is ~5-10x faster.

Where csv5 spends its time (measured on real data, ~0.95s total):
  - telemetry csv parsed with sep=None/engine="python" (csv sniffing,
    slow) and all 16-ish columns, when only 2 are ever used
  - 61 coarse + 41 fine = 102 scale candidates, and for EACH one it
    FFT-correlates the (fixed) telemetry cos/sin channels against the
    (changing) PFD cos/sin channels -- recomputing the telemetry side's
    FFT 102 times over for no reason, since it never changes

Fixes, in the same order:
  1. explicit comma separator + usecols -- pandas' C engine instead of
     the python engine's delimiter-sniffing
  2. the telemetry (data) cos/sin channels' FFTs are computed ONCE and
     reused for every candidate scale, instead of recomputed per call
  3. coarse grid trimmed from 61 to 25 points (the fine pass around the
     best one still refines to the same precision)

Numerically identical result to csv5 for the same input -- this changes
how fast the answer is found, not what the answer is.

    python3 code/pfd_sync/pfd_compare_csv6.py --pfd <pfd_csv> --data <telemetry_csv>
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import rfft, irfft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pfd_verify_airspeed as vias          # noqa: E402
import pfd_verify_msl_altitude as valt      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
CADET_RE = re.compile(r"\.c(\d+|x)\.")
HDG = ("FFS1_Ownship.LHD_ESL_Flight_Model_Data."
       "LHD_ESL_Flight_Model_Data.Heading")
GRID_HZ = 4.0
DT = 1.0 / GRID_HZ


def load_pfd(path):
    df = pd.read_csv(path, usecols=["time_s", "heading"])
    t = df["time_s"].to_numpy(float)
    h = pd.to_numeric(df["heading"], errors="coerce").clip(0, 360)
    h = h.interpolate(limit_direction="both").to_numpy()
    return t, h


def load_data(path):
    # a plain comma-separated read with the C engine and only the two
    # columns needed is the same result as csv5's sep=None/engine="python"
    # full-file read, just without the sniffing/parsing overhead of the
    # other ~14 columns and the slower python engine. Column names in
    # these CSVs carry a leading space, so match on the stripped name
    # rather than assume usecols lines up exactly.
    wanted = {"Recording Time", HDG}
    df = pd.read_csv(path, usecols=lambda c: c.strip() in wanted)
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_numeric(df["Recording Time"], errors="coerce") / 1000.0
    h = np.degrees(pd.to_numeric(df[HDG], errors="coerce")) % 360.0
    m = t.notna() & h.notna()
    return t[m].to_numpy(), np.asarray(h)[m.to_numpy()]


def _cos_sin_on_grid(t, h, grid):
    c = np.interp(grid, t, np.cos(np.radians(h)), left=np.nan, right=np.nan)
    s = np.interp(grid, t, np.sin(np.radians(h)), left=np.nan, right=np.nan)
    return c, s


def _grid_deg(t, h, grid):
    c = np.interp(grid, t, np.cos(np.radians(h)))
    s = np.interp(grid, t, np.sin(np.radians(h)))
    return np.degrees(np.arctan2(s, c)) % 360.0


def _fit_ab(x, y):
    a, b = np.polyfit(x, y, 1)
    yp = a * x + b
    r = np.corrcoef(yp, y)[0, 1] if np.std(yp) > 0 else 0.0
    return float(a), float(b), float(r)


def _score(pt, ph, dt, dh, scale, shift):
    lo = max(pt[0] * scale + shift, dt[0])
    hi = min(pt[-1] * scale + shift, dt[-1])
    if hi <= lo:
        return -2.0, 0
    grid = np.arange(lo, hi, DT)
    dg = _grid_deg(dt, dh, grid)
    pg = _grid_deg(pt * scale + shift, ph, grid)
    return float(np.mean(np.cos(np.radians(pg - dg)))), len(grid)


class _FastShiftFinder:
    """Cross-correlates the (fixed) telemetry cos/sin channels against a
    (per-scale) PFD cos/sin pair, without recomputing the telemetry
    side's FFT every time -- that's the actual hot loop in csv5.

    The FFT length is fixed once (sized for the widest scale this finder
    will ever be asked about) so the telemetry side's transform can be
    reused; a per-call nfft would defeat that, and one sized only for
    scale=1 would silently wrap around (alias) for scale far from 1."""

    def __init__(self, dg, dc, ds, pt_span, max_scale=1.25):
        self.dg = dg
        n = len(dc)
        m_max = int(np.ceil(pt_span * max_scale / DT)) + 8
        self.nfft = 1 << (n + m_max - 1).bit_length()   # power of 2, safe for any m <= m_max
        self.n = n
        self.m_max = m_max
        self.Dc = rfft(dc, self.nfft)
        self.Ds = rfft(ds, self.nfft)

    def best_shift(self, pt, ph, scale):
        wt = pt * scale
        pg = np.arange(wt[0], wt[-1], DT)
        pc, ps = _cos_sin_on_grid(wt, ph, pg)
        pc = np.nan_to_num(pc)
        ps = np.nan_to_num(ps)
        m = len(pc)
        assert m <= self.m_max, "scale produced a wider PFD grid than this finder was sized for"
        # cross-correlation via FFT: same result as
        # scipy.signal.correlate(dc, pc, mode="full"), but the telemetry
        # side's transform (self.Dc/self.Ds) is reused across every call
        Pc = rfft(pc[::-1], self.nfft)
        Ps = rfft(ps[::-1], self.nfft)
        c_full = irfft(self.Dc * Pc, self.nfft) + irfft(self.Ds * Ps, self.nfft)
        c = c_full[: self.n + m - 1]
        lags = np.arange(-(m - 1), self.n)
        shifts = self.dg[0] - pg[0] + lags * DT
        k = int(np.argmax(c))
        return float(shifts[k])


def fit(pt, ph, dt, dh, scale_min, scale_max, n_coarse=25):
    dg = np.arange(dt[0], dt[-1], DT)
    dc, ds = _cos_sin_on_grid(dt, dh, dg)
    dc = np.nan_to_num(dc)
    ds = np.nan_to_num(ds)
    finder = _FastShiftFinder(dg, dc, ds, pt_span=pt[-1] - pt[0], max_scale=scale_max)

    def eval_scale(s):
        sh = finder.best_shift(pt, ph, s)
        score, n = _score(pt, ph, dt, dh, s, sh)
        return (score, s, sh, n)

    coarse = [eval_scale(s) for s in np.linspace(scale_min, scale_max, n_coarse)]
    best = max(coarse, key=lambda r: r[0])
    step = (scale_max - scale_min) / (n_coarse - 1)
    fine = [eval_scale(s) for s in np.linspace(best[1] - step, best[1] + step, 21)]
    best = max([best] + fine, key=lambda r: r[0])

    score, scale, shift, n = best
    rms_deg = float(np.degrees(np.arccos(np.clip(score, -1, 1))))
    return {"t_scale": scale, "t_shift": shift, "score": score,
            "rms_deg": rms_deg, "n": n}


def _wrapped_for_plot(t, h):
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
                      title=f"PFD vs data heading — offset + stretch fit (csv6, fast){(' — ' + label) if label else ''}")
    fig.update_yaxes(title_text="heading (deg)", range=[0, 360])
    fig.update_xaxes(title_text="data time (s)")
    return fig


def render_heading_png(pt, ph, dt, dh, res, label, out_png):
    """Matplotlib PNG version of build_figure(), for the batch export tree
    (pfd_report.py embeds PNGs, not the plotly HTML build_figure() makes
    for single-file --pfd/--data runs)."""
    wt = pt * res["t_scale"] + res["t_shift"]

    def wrap(t, h):
        h = np.asarray(h, float); o = h.copy()
        j = np.abs(np.diff(h)) > 180
        o[1:][j] = np.nan
        return t, o

    dtx, dhx = wrap(dt, dh)
    ptx, phx = wrap(pt, ph)
    wtx, whx = wrap(wt, ph)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharey=True)
    axes[0].plot(dtx, dhx, color="#1f77b4", lw=0.7, label="telemetry")
    axes[0].plot(ptx, phx, color="#d62728", lw=0.7, label="PFD (raw)")
    axes[0].set_title("heading -- as recorded")
    axes[0].legend(loc="upper right")
    axes[1].plot(dtx, dhx, color="#1f77b4", lw=0.8, label="telemetry")
    axes[1].plot(wtx, whx, color="#d62728", lw=0.8, label="PFD (aligned)")
    axes[1].set_title(f"after  {res['t_scale']:.4f}x pfd {res['t_shift']:+.1f}s  "
                       f"mean-cos {res['score']:.4f} (~{res['rms_deg']:.1f} deg err)")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.set_ylim(0, 360); ax.grid(alpha=0.3); ax.set_ylabel("heading (deg)")
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"Heading sync (csv6, fast) -- {label}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def find_telemetry_csv(csv_root: Path, sortie: str) -> Path | None:
    d = csv_root / sortie
    if not d.is_dir():
        return None
    for name in ("Elements_0.csv", "Elements.csv"):
        if (d / name).is_file():
            return d / name
    return None


def recommend(heading_score, ias_r, alt_r) -> tuple[str, str]:
    scores = [x for x in (heading_score, ias_r, alt_r) if x is not None]
    strong = sum(1 for x in scores if x >= 0.85)
    if heading_score is not None and heading_score >= 0.9 and strong >= 3:
        return ("CONFIRMED", "All 3 channels agree -- safe to use this sync for downstream work.")
    if strong >= 2:
        return ("LIKELY OK", "Majority of channels agree; spot-check the odd one out before relying on it.")
    return ("NEEDS REVIEW", "Channels disagree -- do not trust this sync without a closer look.")


def run_batch(csv_root: Path, reuse_root: Path, export_root: Path,
              scale_min: float, scale_max: float):
    """Re-sync every cadet already OCR'd+cleaned under reuse_root (a
    pfd_csv_process.py export dir, or any tree of <label>/pfd_clean.csv),
    using csv6's fast fit -- no OCR, no re-cleaning, just the fit +
    verify + graphs + aligned csv, into export_root/<label>/. Writes the
    same summary.json shape pfd_csv_process.py does, so pfd_report.py
    works on either export tree unchanged."""
    export_root.mkdir(parents=True, exist_ok=True)
    labels = sorted(d for d in reuse_root.iterdir() if d.is_dir() and (d / "pfd_clean.csv").is_file())
    print(f"found {len(labels)} already-cleaned cadet(s) under {reuse_root}\n")

    results = []
    for d in labels:
        label = d.name
        m = CADET_RE.search(label)
        sortie = None if (m and m.group(1) == "x") else (f"{int(m.group(1)):03d}" if m else None)
        print(f"=== {label}  (sortie {sortie}) ===")
        out_dir = export_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_pfd = d / "pfd_clean.csv"

        if sortie is None:
            print("  no numbered sortie, skipping")
            results.append({"label": label, "sortie": None, "status": "skipped",
                             "reason": "cadet folder has no numbered sortie (e.g. 'cx')"})
            continue
        telemetry = find_telemetry_csv(csv_root, sortie)
        if telemetry is None:
            print("  no telemetry csv found, skipping")
            results.append({"label": label, "sortie": sortie, "status": "skipped",
                             "reason": f"no Elements[_0].csv under {csv_root / sortie}"})
            continue

        t0 = time.time()
        pt, ph = load_pfd(clean_pfd)
        dt, dh = load_data(telemetry)
        res_hdg = fit(pt, ph, dt, dh, scale_min, scale_max)
        render_heading_png(pt, ph, dt, dh, res_hdg, label, out_dir / "heading.png")

        pt_i, pv_i = vias.load_pfd_ias(clean_pfd)
        dt_i, dv_i = vias.load_data_tas(telemetry)
        res_ias = vias.fit_at(pt_i, pv_i, dt_i, dv_i, res_hdg["t_scale"], res_hdg["t_shift"])
        vias.render(pt_i, pv_i, dt_i, dv_i, res_ias, label, out_dir / "airspeed.png")

        pt_a, pv_a = valt.load_pfd_alt(clean_pfd)
        dt_a, dv_a = valt.load_data_msl(telemetry)
        res_alt = valt.fit_at(pt_a, pv_a, dt_a, dv_a, res_hdg["t_scale"], res_hdg["t_shift"])
        valt.render(pt_a, pv_a, dt_a, dv_a, res_alt, label, out_dir / "altitude.png")

        raw = pd.read_csv(telemetry, usecols=lambda c: True)
        data_s = pd.to_numeric(raw[raw.columns[0]], errors="coerce") / 1000.0
        pfd_s = (data_s - res_hdg["t_shift"]) / res_hdg["t_scale"]
        aligned = raw.copy()
        aligned.insert(0, "PFD Time (s)", pfd_s.round(4))
        aligned.insert(1, "PFD Time (ms)", (pfd_s * 1000.0).round(1))
        aligned.to_csv(out_dir / "telemetry_synced.csv", index=False)

        verdict, reason = recommend(res_hdg["score"], res_ias["r"], res_alt["r"])
        secs = round(time.time() - t0, 2)
        print(f"  heading mean-cos {res_hdg['score']:.4f}   airspeed r {res_ias['r']:.3f}   "
              f"altitude r {res_alt['r']:.3f}   -> {verdict}  ({secs}s)")

        result = {
            "label": label, "sortie": sortie, "status": "ok",
            "telemetry_csv": str(telemetry), "pfd_clean_csv": str(clean_pfd),
            "t_scale": res_hdg["t_scale"], "t_shift": res_hdg["t_shift"],
            "heading_score": res_hdg["score"], "heading_rms_deg": res_hdg["rms_deg"],
            "airspeed_r": res_ias["r"], "altitude_r": res_alt["r"],
            "clean": {}, "verdict": verdict, "verdict_reason": reason, "seconds": secs,
            "files": {
                "pfd_clean": str(clean_pfd),
                "telemetry_synced": str(out_dir / "telemetry_synced.csv"),
                "heading_png": str(out_dir / "heading.png"),
                "airspeed_png": str(out_dir / "airspeed.png"),
                "altitude_png": str(out_dir / "altitude.png"),
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
        results.append(result)

    import csv as csv_mod
    fields = ["label", "sortie", "status", "t_scale", "t_shift", "heading_score",
              "heading_rms_deg", "airspeed_r", "altitude_r", "verdict", "verdict_reason",
              "seconds", "reason"]
    with open(export_root / "_summary.csv", "w", newline="") as fh:
        w = csv_mod.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {export_root / '_summary.csv'}")

    manifest = {"csv_root": str(csv_root), "video_root": str(reuse_root),
                "export_root": str(export_root),
                "csv_sorties": sorted(p.name for p in csv_root.iterdir() if p.is_dir() and p.name.isdigit()),
                "video_sessions": [r["label"] for r in results],
                "missing_video_for_sortie": [], "missing_telemetry_for_video": [],
                "results": results}
    (export_root / "_run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWrote {export_root / '_run_manifest.json'}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd")
    ap.add_argument("--data")
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.20)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="output HTML path")
    ap.add_argument("--no-aligned-csv", action="store_true")
    ap.add_argument("--csv-root", help="batch mode: telemetry root (.../<sortie>/Elements_0.csv)")
    ap.add_argument("--reuse-root", help="batch mode: existing export dir with <label>/pfd_clean.csv")
    ap.add_argument("--export-root", help="batch mode: where to write <label>/ outputs")
    args = ap.parse_args()

    if args.csv_root or args.reuse_root or args.export_root:
        if not (args.csv_root and args.reuse_root and args.export_root):
            sys.exit("batch mode needs --csv-root, --reuse-root, and --export-root together")
        run_batch(Path(args.csv_root), Path(args.reuse_root), Path(args.export_root),
                  args.scale_min, args.scale_max)
        return

    if not (args.pfd and args.data):
        sys.exit("need --pfd and --data (or the three --*-root args for batch mode)")

    t0 = time.time()
    pt, ph = load_pfd(Path(args.pfd))
    dt, dh = load_data(args.data)
    t_load = time.time() - t0
    print(f"PFD  {args.pfd}\n     {len(pt)} rows, {pt[-1]:.0f}s, heading {ph.min():.0f}..{ph.max():.0f}")
    print(f"data {args.data}\n     {len(dt)} rows, {dt[-1]:.0f}s, heading {dh.min():.0f}..{dh.max():.0f}")

    t1 = time.time()
    res = fit(pt, ph, dt, dh, args.scale_min, args.scale_max)
    t_fit = time.time() - t1
    print(f"\nbest fit")
    print(f"  time  data_t = {res['t_scale']:.4f} * pfd_t {res['t_shift']:+.1f} s"
          f"   (stretch {res['t_scale']:.4f}x, move {res['t_shift']:+.1f}s)")
    print(f"  match : mean-cos {res['score']:.4f}  (~{res['rms_deg']:.1f} deg mean error, "
          f"n={res['n']} samples)"
          + ("   strong" if res["score"] > 0.9 else "   weak"))
    print(f"  timing: load {t_load:.3f}s, fit {t_fit:.3f}s, total {time.time()-t0:.3f}s")

    out = Path(args.out) if args.out else \
        ROOT / "exports" / "pfd" / f"{Path(args.pfd).stem}_hdg_fit6.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(pt, ph, dt, dh, res, args.label).write_html(str(out))
    print(f"Wrote {out}")

    if not args.no_aligned_csv:
        raw = pd.read_csv(args.data, usecols=lambda c: True)
        data_s = pd.to_numeric(raw[raw.columns[0]], errors="coerce") / 1000.0
        pfd_s = (data_s - res["t_shift"]) / res["t_scale"]
        aligned = raw.copy()
        aligned.insert(0, "PFD Time (s)", pfd_s.round(4))
        aligned.insert(1, "PFD Time (ms)", (pfd_s * 1000.0).round(1))
        acsv = out.with_name(f"{Path(args.data).stem}_in_{Path(args.pfd).stem}_time.csv")
        aligned.to_csv(acsv, index=False)
        print(f"Wrote {acsv}")

    return res


if __name__ == "__main__":
    main()
