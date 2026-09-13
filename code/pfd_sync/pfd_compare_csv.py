"""Find the drills in a check-ride and time them, using the sim data CSV
as the source of truth and the PFD-OCR CSV for cross-reference.

What it does
------------
1. Loads the data CSV, resamples to a uniform grid, derives vertical
   speed, bank, heading rate and collective rate.
2. Segments the flight into activity blocks separated by quiet / frozen
   gaps (sim pauses between exercises), splitting long blocks at internal
   lulls, and classifies each block by its signature
   (climb / descent / steep turn / low-speed / turn / maneuvering).
3. Tries to line the PFD video up with the data by cross-correlating the
   signals they share -- PFD pi vs collective, PFD heading vs heading --
   globally and per quarter, and also finds a *local* offset for each
   drill (peak of pi-vs-collective correlation in a window around it).
   If those local offsets do not agree, the video timeline is not linear
   in sim time (pauses / seeks in the review tool) and only the data-time
   column is trustworthy.
4. Prints the drill table and writes a Plotly HTML: data signals with the
   drills shaded, plus the PFD traces before and after the global shift.

Usage
-----
    python3 code/pfd_sync/pfd_compare_csv.py
    python3 code/pfd_sync/pfd_compare_csv.py --pfd other_pfd.csv --data other/Elements_0.csv
    cd "/home/ilan/Documents/flight school/drill_compare" && python3 code/pfd_sync/pfd_compare_csv.py
python3 code/pfd_sync/pfd_compare_csv.py \
  --pfd "exports/pfd/s1 c6 pfd sorek bigeyes blk hair_pfd.csv" \
  --data "/home/ilan/Documents/flight school/data/csv_data/1_26/CSV_3.6.26_check1/006/Elements_0.csv"

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

FM = "FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data."
CP = "FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.CollectivePosition"
HOT = "FFS1_Ownship.ESL_LHD_Landing_detection.ESL_LHD_Landing_detection.Point_1_HOT"
GRID_HZ = 5.0
DT = 1.0 / GRID_HZ

M2FT = 3.28084
MS2KT = 1.94384


# ----------------------------------------------------------------------------- data

def _smooth(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()


def load_data(path):
    # recordings are usually TAB-delimited (Elements_0.csv) but some are comma;
    # let the engine sniff. Headers carry a leading space -> strip.
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    t_all = pd.to_numeric(df["Recording Time"], errors="coerce") / 1000.0
    # Point_1_HOT (ground height) is logged faster and sparsely elsewhere;
    # fill it before deriving AGL the way the app does.
    hot = pd.to_numeric(df[HOT], errors="coerce").interpolate(limit_direction="both")
    cols = [FM + "AltitudeMSL", FM + "Heading", FM + "TrueAirSpeed", FM + "Roll", CP]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    sub["t"] = t_all
    # sparse channels (collective, heading, ...) carry blanks between samples
    sub = sub[sub["t"].notna()].sort_values("t")
    sub[cols] = sub[cols].interpolate(limit_direction="both")
    sub["agl"] = (sub[FM + "AltitudeMSL"] - hot.reindex(sub.index)) * M2FT
    sub["agl"] = sub["agl"].interpolate(limit_direction="both")
    sub = sub.dropna().reset_index(drop=True)
    t = sub["t"].to_numpy()
    g = np.arange(t[0], t[-1], DT)
    d = {
        "t": g,
        "alt": np.interp(g, t, sub["agl"]),                       # feet AGL
        "msl": np.interp(g, t, sub[FM + "AltitudeMSL"]) * M2FT,   # feet MSL
        "tas": np.interp(g, t, sub[FM + "TrueAirSpeed"]) * MS2KT,
        "roll": np.degrees(np.interp(g, t, sub[FM + "Roll"])),
        "coll": np.interp(g, t, sub[CP]),
        "hdg": np.interp(g, t, np.degrees(np.unwrap(sub[FM + "Heading"].to_numpy()))),
    }
    msl_f = pd.Series(d["msl"]).rolling(9, center=True, min_periods=1).median().to_numpy()
    d["vs"] = np.clip(_smooth(np.gradient(msl_f, DT), 15) * 60.0, -4000, 4000)  # ft/min
    d["hdg_rate"] = _smooth(np.gradient(d["hdg"], DT), 15)           # deg/s
    d["coll_rate"] = np.gradient(_smooth(d["coll"], 9), DT)
    d["activity"] = _smooth(
        np.abs(d["vs"]) / 300 + np.abs(d["roll"]) / 10
        + np.abs(d["hdg_rate"]) / 2 + np.abs(d["coll_rate"]) * 40, 25)
    return d


# ----------------------------------------------------------------------- drills

def _split_long(a, b, act, g, thr, max_len=300, min_lull=12):
    if g[b] - g[a] <= max_len:
        return [(a, b)]
    seg = act[a:b + 1]
    lull = seg < thr * 1.4
    runs, i = [], 0
    while i < len(lull):
        if lull[i]:
            j = i
            while j < len(lull) and lull[j]:
                j += 1
            if (j - i) * DT >= min_lull:
                runs.append((i + j) // 2 + a)
            i = j
        else:
            i += 1
    cuts = sorted({a, *runs, b})
    segs = [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)
            if g[cuts[k + 1]] - g[cuts[k]] >= 35]
    # anything still very long: chop into roughly-equal pieces
    out = []
    for s, e in segs or [(a, b)]:
        span = g[e] - g[s]
        n = max(1, int(np.ceil(span / max_len)))
        for k in range(n):
            out.append((s + (e - s) * k // n, s + (e - s) * (k + 1) // n))
    return out


def find_drills(d, gap=20.0, min_len=25.0):
    g, act = d["t"], d["activity"]
    thr = np.percentile(act, 35) * 1.3
    idx = np.where(act > thr)[0]
    blocks = []
    if len(idx):
        s = p = idx[0]
        for i in idx[1:]:
            if g[i] - g[p] > gap:
                blocks.append((s, p))
                s = i
            p = i
        blocks.append((s, p))
    spans = []
    for a, b in blocks:
        if g[b] - g[a] < min_len:
            continue
        spans += _split_long(a, b, act, g, thr)

    drills = []
    for a, b in spans:
        m = slice(a, b + 1)
        alt, vs, tas, roll = d["alt"][m], d["vs"][m], d["tas"][m], d["roll"][m]
        dh = d["hdg"][b] - d["hdg"][a]
        bank = np.abs(roll)
        frac_steep = np.mean(bank > 25)
        kind = "maneuvering"
        if tas.min() < 45 and alt.mean() < 900:
            kind = "low-speed / quick-stop"
        elif frac_steep > 0.20:
            kind = "steep turns" if abs(dh) < 300 else "steep 360s"
        elif abs(dh) >= 300:
            kind = "circuit / pattern"
        elif vs.min() < -900 and np.mean(vs < -400) > 0.4:
            kind = "descent / autorotation"
        elif vs.max() > 800 and np.mean(vs > 350) > 0.4:
            kind = "climb"
        elif abs(dh) >= 80:
            kind = "turn"
        drills.append({
            "start": float(g[a]), "end": float(g[b]), "dur": float(g[b] - g[a]),
            "kind": kind,
            "alt_lo": float(alt.min()), "alt_hi": float(alt.max()),
            "vs_min": float(vs.min()), "vs_max": float(vs.max()),
            "tas_min": float(tas.min()), "bank_max": float(bank.max()),
            "dhdg": float(dh),
        })

    # merge neighbours of the same kind that are only a short gap apart
    merged = []
    for d in drills:
        if merged and d["kind"] == merged[-1]["kind"] \
                and d["start"] - merged[-1]["end"] <= 25 \
                and merged[-1]["end"] - merged[-1]["start"] < 240:
            m = merged[-1]
            m["end"] = d["end"]
            m["dur"] = m["end"] - m["start"]
            m["alt_lo"] = min(m["alt_lo"], d["alt_lo"])
            m["alt_hi"] = max(m["alt_hi"], d["alt_hi"])
            m["vs_min"] = min(m["vs_min"], d["vs_min"])
            m["vs_max"] = max(m["vs_max"], d["vs_max"])
            m["tas_min"] = min(m["tas_min"], d["tas_min"])
            m["bank_max"] = max(m["bank_max"], d["bank_max"])
            m["dhdg"] += d["dhdg"]
        else:
            merged.append(dict(d))
    return merged


# -------------------------------------------------------------------- alignment

def _clean_pfd_col(s, lo, hi, out_win=15):
    s = pd.to_numeric(s, errors="coerce").clip(lo, hi)
    med = s.rolling(61, center=True, min_periods=1).median()
    s = s.where((s - med).abs() <= 0.08 * (hi - lo), med)
    return s.rolling(out_win, center=True, min_periods=1).median() \
            .interpolate(limit_direction="both").to_numpy()


def load_pfd(path):
    df = pd.read_csv(path)
    return {
        "t": df["time_s"].to_numpy(),
        "alt": _clean_pfd_col(df["altitude"], 0, 3000),
        "ra": _clean_pfd_col(df["ra"], 0, 1600),
        "hdg": _clean_pfd_col(df["heading"], 1, 360),
        "pi": _clean_pfd_col(df["pi"], 0, 110),
    }


def _norm(x):
    x = x - np.mean(x)
    sd = x.std()
    return x / sd if sd else x


def xcorr_offset(pt, pv, dt_t, dv, heading=False, deriv=True, max_off=1200):
    gp = np.arange(pt[0], pt[-1], DT)
    gd = np.arange(dt_t[0], dt_t[-1], DT)
    a = np.interp(gd, dt_t, dv)
    b = np.interp(gp, pt, pv)
    if heading:
        a = np.degrees(np.unwrap(np.radians(a)))
        b = np.degrees(np.unwrap(np.radians(b)))
    if deriv:
        a, b = np.gradient(a), np.gradient(b)
    a, b = _norm(a), _norm(b)
    c = signal.correlate(a, b, mode="full")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    off = (gd[0] - gp[0]) + lags * DT
    keep = np.abs(off) <= max_off
    c, off = c[keep], off[keep]
    k = int(np.argmax(c))
    return float(off[k]), float(c[k] / min(len(a), len(b)))


def quarter_offsets(pfd, data, key_p, key_d, heading):
    t0, t1 = pfd["t"][0], pfd["t"][-1]
    res = []
    for i in range(4):
        a, b = t0 + i * (t1 - t0) / 4, t0 + (i + 1) * (t1 - t0) / 4
        m = (pfd["t"] >= a) & (pfd["t"] < b)
        res.append(xcorr_offset(pfd["t"][m], pfd[key_p][m],
                                data["t"], data[key_d], heading=heading))
    return res


# ------------------------------------------------------------------------ output

def print_report(drills, glob, quarters, rec, conf):
    def mmss(x):
        return f"{int(x)//60:d}:{int(x) % 60:02d}"

    print(f"\n{'#':>2}  {'type':22} {'data time':>15} {'dur':>5}  {'AGL ft':>9}  notes")
    print("-" * 96)
    for i, d in enumerate(drills, 1):
        notes = []
        if d["vs_min"] < -700:
            notes.append(f"VS {d['vs_min']:.0f}")
        if d["vs_max"] > 700:
            notes.append(f"VS +{d['vs_max']:.0f}")
        if d["bank_max"] > 25:
            notes.append(f"bank {d['bank_max']:.0f}")
        if d["tas_min"] < 45:
            notes.append(f"TAS {d['tas_min']:.0f}")
        if abs(d["dhdg"]) > 120:
            notes.append(f"dHdg {d['dhdg']:+.0f}")
        print(f"{i:>2}  {d['kind']:22} {mmss(d['start']):>7}-{mmss(d['end']):<7} "
              f"{d['dur']:>4.0f}s  {d['alt_lo']:>4.0f}-{d['alt_hi']:<4.0f}  "
              f"{', '.join(notes)}")

    print("\nalignment (offset to ADD to PFD time so it matches data time):")
    for name, (o, s) in glob.items():
        print(f"  {name:22} {o:+8.1f} s  score {s:.3f}")
    for name, qs in quarters.items():
        print(f"  {name:22} by quarter: " +
              "  ".join(f"{o:+.0f}({s:.2f})" for o, s in qs))
    print(f"\nRECOMMENDED OFFSET: {rec:+.1f} s  ({rec*60:+.0f} PFD frames)")
    print(f"  {conf}")


PLOT_HZ = 2.0                       # common grid both sources are put on for the plot


def _grid(t, v, hz, heading=False):
    """Resample onto a uniform grid at `hz` and scale amplitude to 0..1
    using the 2nd..98th percentile range (ignores lone OCR spikes)."""
    g = np.arange(t[0], t[-1], 1.0 / hz)
    y = np.interp(g, t, v)
    if heading:
        y = np.degrees(np.unwrap(np.radians(y))) % 360.0
    lo, hi = np.nanpercentile(y, 2), np.nanpercentile(y, 98)
    y = np.clip((y - lo) / (hi - lo), 0.0, 1.0) if hi > lo else y * 0.0
    return g, y


def build_figure(pfd, data, drills, offset, conf):
    # (PFD key, data key, label, is-heading)
    rows = [("ra", "alt", "ra  /  AGL", False),
            ("alt", "msl", "altitude  /  MSL", False),
            ("hdg", "hdg", "heading", True),
            ("pi", "coll", "pi  /  collective", False)]
    fig = make_subplots(
        rows=len(rows), cols=2, shared_xaxes=True,
        horizontal_spacing=0.05, vertical_spacing=0.05,
        column_titles=["overlaid as recorded",
                       f"PFD shifted {offset:+.0f}s"],
        row_titles=[lbl for _, _, lbl, _ in rows])
    for ri, (pk, dk, _, is_hdg) in enumerate(rows, 1):
        gd, yd = _grid(data["t"], data[dk], PLOT_HZ, heading=is_hdg)
        gp, yp = _grid(pfd["t"], pfd[pk], PLOT_HZ, heading=is_hdg)
        for ci, shift in ((1, 0.0), (2, offset)):
            fig.add_trace(go.Scatter(
                x=gd, y=yd, line=dict(color="#1f77b4", width=1),
                name="data", legendgroup="data",
                showlegend=(ri == 1 and ci == 1)), row=ri, col=ci)
            fig.add_trace(go.Scatter(
                x=gp + shift, y=yp, line=dict(color="#d62728", width=1),
                name="PFD", legendgroup="pfd",
                showlegend=(ri == 1 and ci == 1)), row=ri, col=ci)
        for d in drills:
            fig.add_vrect(x0=d["start"], x1=d["end"], row=ri, col=1,
                          fillcolor="green", opacity=0.06, line_width=0)
    fig.update_layout(
        height=250 * len(rows), width=1500, template="plotly_white",
        legend=dict(orientation="h"),
        title=f"PFD (red) vs data (blue) — both resampled to {PLOT_HZ:g} Hz, "
              f"each trace scaled 0–1<br><sub>{conf}</sub>")
    fig.update_yaxes(range=[-0.05, 1.05])
    fig.update_xaxes(title_text="time (s)", row=len(rows))
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfd", default=str(DEF_PFD))
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--gap", type=float, default=20.0,
                    help="quiet seconds that separate two drills (default 20)")
    ap.add_argument("--offset", type=float,
                    help="seconds to add to PFD time in the 'shifted' plot; "
                         "overrides the auto value so you can dial it in by eye")
    ap.add_argument("--out", help="output HTML path")
    args = ap.parse_args()

    data = load_data(args.data)
    pfd = load_pfd(Path(args.pfd))
    print(f"data {args.data}\n     {data['t'][-1]:.0f}s")
    print(f"PFD  {args.pfd}\n     {pfd['t'][-1]:.0f}s")

    drills = find_drills(data, gap=args.gap)

    glob = {
        "ra vs AGL": xcorr_offset(pfd["t"], pfd["ra"], data["t"], data["alt"]),
        "alt vs MSL": xcorr_offset(pfd["t"], pfd["alt"], data["t"], data["msl"]),
        "pi vs collective": xcorr_offset(pfd["t"], pfd["pi"], data["t"], data["coll"]),
        "heading vs heading": xcorr_offset(pfd["t"], pfd["hdg"], data["t"], data["hdg"],
                                           heading=True),
    }
    quarters = {
        "ra vs AGL": quarter_offsets(pfd, data, "ra", "alt", False),
        "pi vs collective": quarter_offsets(pfd, data, "pi", "coll", False),
        "heading vs heading": quarter_offsets(pfd, data, "hdg", "hdg", True),
    }
    offs = np.array([v[0] for v in glob.values()])
    scores = np.array([v[1] for v in glob.values()])
    spread = float(offs.max() - offs.min())
    best_name = list(glob)[int(np.argmax(scores))]
    rec = glob[best_name][0]
    q_all = np.array([o for qs in quarters.values() for o, _ in qs])
    q_spread = float(q_all.max() - q_all.min())
    if spread <= 10 and q_spread <= 60:
        conf = f"signals agree (spread {spread:.0f}s, quarters {q_spread:.0f}s) -- offset is reliable"
    else:
        conf = (f"signals disagree (global spread {spread:.0f}s, quarter spread "
                f"{q_spread:.0f}s): PFD video time is NOT linear in sim time -- "
                f"trust the data-time column, not the video mapping")

    print_report(drills, glob, quarters, rec, conf)

    shift = args.offset if args.offset is not None else rec
    if args.offset is not None:
        conf = f"manual --offset {shift:+.0f}s"
    out = Path(args.out) if args.out else \
        ROOT / "exports" / "pfd" / f"{Path(args.pfd).stem}_drills.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(pfd, data, drills, shift, conf).write_html(str(out))

    csv_out = out.with_suffix(".csv")
    pd.DataFrame(drills).to_csv(csv_out, index=False)
    print(f"\nWrote {out}\nWrote {csv_out}")


if __name__ == "__main__":
    main()
