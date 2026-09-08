#!/usr/bin/env python3
"""Build a cross-sortie dashboard from the controls-biomarker _summary.json files.

Reads every ``exports/controls/<sortie>/_summary.json`` produced by ``run_all.py``
and writes a single standalone HTML file (no external deps) with one heatmap
table per biomarker: rows = sorties, columns = channel x metric, cells shaded
green (good) -> red (poor) per column. Data-quality problems (degenerate
correction-gain, blown-up overshoot ratios) are flagged inline.

    python3 code/controls_biomarkers/dashboard.py
    python3 code/controls_biomarkers/dashboard.py --open
"""
from __future__ import annotations

import argparse
import html
import json
import math
import webbrowser
from pathlib import Path

CONTROLS_DIR = Path(__file__).resolve().parent.parent.parent / "exports" / "controls"
OUT = CONTROLS_DIR / "dashboard.html"
CHANNELS = ["collective", "lateral", "longitudinal"]

# (biomarker, metric, label, direction)  direction: +1 higher is better, -1 lower
# is better, 0 neutral (shown but not shaded).
METRICS = [
    ("controls_precision", "precision_score_0_1", "precision", +1),
    ("controls_precision", "median_hold_std", "hold std", -1),
    ("controls_smoothness", "rate_rms", "rate rms", -1),
    ("controls_smoothness", "jerk_rms", "jerk rms", -1),
    ("controls_smoothness", "oscillations_per_min", "osc/min", -1),
    ("controls_efficiency", "mean_efficiency", "mean eff", +1),
    ("controls_efficiency", "median_efficiency", "med eff", +1),
    ("controls_correction_count", "mean_corrections", "mean corr", -1),
    ("controls_correction_count", "clean_single_input_rate", "clean input", +1),
    ("controls_correction_count", "max_corrections", "max corr", -1),
    ("controls_overshoot", "mean_crossings", "crossings", -1),
    ("controls_overshoot", "clean_arrival_rate", "clean arrival", +1),
    ("controls_overshoot", "mean_overshoot_ratio", "overshoot ratio", -1),
    ("controls_correction_gain", "gain", "gain", 0),
    ("controls_correction_gain", "r", "r", 0),
    ("controls_anticipation", "mean_anticipation_s", "anticip s", +1),
    ("controls_anticipation", "anticipatory_share", "anticip share", +1),
    ("controls_anticipation", "reactive_share", "reactive share", -1),
    ("controls_stabilization_time", "mean_stabilization_s", "stab mean s", -1),
    ("controls_stabilization_time", "p90_stabilization_s", "stab p90 s", -1),
    ("controls_aggressiveness", "median_aggressiveness_index", "aggr index", 0),
    ("controls_aggressiveness", "aggressive_share", "aggr share", 0),
    ("controls_aggressiveness", "conservative_share", "consv share", 0),
]

# Cells whose magnitude exceeds this are treated as sensor/estimator artifacts:
# shown with a warning marker and excluded from the colour scale.
SANE_MAX = {"mean_overshoot_ratio": 100.0}


def load_summaries() -> list[dict]:
    out = []
    for d in sorted(CONTROLS_DIR.iterdir()):
        f = d / "_summary.json" if d.is_dir() else None
        if f and f.exists():
            out.append(json.loads(f.read_text()))
    return out


def get(summary: dict, bm: str, ch: str, metric: str):
    try:
        return summary["biomarkers"][bm]["per_channel"][ch][metric]
    except (KeyError, TypeError):
        return None


def is_bad(metric: str, v) -> bool:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return True
    cap = SANE_MAX.get(metric)
    return cap is not None and abs(v) > cap


def shade(v, lo, hi, direction) -> str:
    """Return a background colour for value v within [lo, hi]."""
    if direction == 0 or v is None or hi is None or lo is None or hi == lo:
        return ""
    frac = (v - lo) / (hi - lo)
    if direction < 0:
        frac = 1.0 - frac
    frac = max(0.0, min(1.0, frac))
    # red (0) -> yellow (.5) -> green (1)
    if frac < 0.5:
        r, g = 214, int(153 + (196 - 153) * frac * 2)
    else:
        r, g = int(214 - (214 - 120) * (frac - 0.5) * 2), 196
    return f"background:rgb({r},{g},120);color:#1a1a1a"


def fmt(v) -> str:
    if v is None:
        return "&mdash;"
    if isinstance(v, float):
        if v != 0 and abs(v) < 0.01:
            return f"{v:.4f}"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.2f}"
    return html.escape(str(v))


def build_table(summaries: list[dict], bm: str) -> str:
    cols = [(m, lbl, dr) for (b, m, lbl, dr) in METRICS if b == bm]
    head = ["<th class='sortie'>sortie</th>"]
    for ch in CHANNELS:
        for _, lbl, _ in cols:
            head.append(f"<th><span class='ch ch-{ch[:3]}'>{ch[:4]}</span><br>{html.escape(lbl)}</th>")

    # per-column (channel x metric) range across sorties, ignoring artifacts
    ranges = {}
    for ch in CHANNELS:
        for m, _, dr in cols:
            vals = [get(s, bm, ch, m) for s in summaries]
            vals = [v for v in vals if v is not None and not is_bad(m, v)]
            ranges[(ch, m)] = (min(vals), max(vals)) if vals else (None, None)

    rows = []
    for s in summaries:
        cells = [f"<td class='sortie'>{html.escape(s['sortie'])}"
                 f"<span class='dur'>{s.get('duration_min','?')}m</span></td>"]
        for ch in CHANNELS:
            for m, _, dr in cols:
                v = get(s, bm, ch, m)
                lo, hi = ranges[(ch, m)]
                if is_bad(m, v):
                    cells.append(f"<td class='bad' title='out of sane range / missing'>"
                                 f"{fmt(v)} &#9888;</td>")
                else:
                    cells.append(f"<td style='{shade(v, lo, hi, dr)}'>{fmt(v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (f"<h2 id='{bm}'>{bm.replace('controls_', '').replace('_', ' ')}</h2>"
            f"<div class='scroll'><table><thead><tr>{''.join(head)}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def quality_notes(summaries: list[dict]) -> str:
    notes = []
    # degenerate correction gain
    allzero = all(
        abs(get(s, "controls_correction_gain", ch, "gain") or 0) < 1e-6
        and (get(s, "controls_correction_gain", ch, "r") or 0) < 0.3
        for s in summaries for ch in CHANNELS
    )
    if allzero:
        notes.append("<b>controls_correction_gain</b> is ~0 with r&lt;0.3 on every "
                     "channel/sortie &mdash; the estimator produces no usable signal "
                     "(degenerate; treat as not implemented).")
    # overshoot blow-ups
    blown = [(s["sortie"], ch, get(s, "controls_overshoot", ch, "mean_overshoot_ratio"))
             for s in summaries for ch in CHANNELS
             if is_bad("mean_overshoot_ratio", get(s, "controls_overshoot", ch, "mean_overshoot_ratio"))]
    if blown:
        items = ", ".join(f"{sid}/{ch} = {val:,.0f}" for sid, ch, val in blown)
        notes.append(f"<b>controls_overshoot.mean_overshoot_ratio</b> blows up "
                     f"(divide-by-near-zero): {items}. Needs a guard/clamp.")
    if not notes:
        return ""
    items = "".join(f"<li>{n}</li>" for n in notes)
    return f"<div class='quality'><h2>data-quality flags</h2><ul>{items}</ul></div>"


def render(summaries: list[dict]) -> str:
    biomarkers = list(dict.fromkeys(b for (b, *_1) in METRICS))
    nav = " · ".join(f"<a href='#{b}'>{b.replace('controls_', '')}</a>" for b in biomarkers)
    tables = "\n".join(build_table(summaries, b) for b in biomarkers)
    sortie_list = ", ".join(f"{s['sortie']} ({s.get('duration_min','?')}m)" for s in summaries)
    src = summaries[0]["source"].rsplit("/", 3)[0] if summaries else ""
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>controls biomarkers &mdash; cross-sortie dashboard</title>
<style>
 body{{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;background:#fafafa}}
 h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:14px;margin:26px 0 8px;text-transform:capitalize}}
 .meta{{color:#666;font-size:12px;margin-bottom:12px}}
 nav{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;font-size:12px;z-index:2}}
 nav a{{color:#2563a8;text-decoration:none;white-space:nowrap}}
 .scroll{{overflow-x:auto}}
 table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}
 th,td{{border:1px solid #ddd;padding:3px 7px;text-align:right}}
 thead th{{background:#f0f0f0;font-weight:600;font-size:11px;vertical-align:bottom}}
 td.sortie,th.sortie{{text-align:left;font-weight:600;position:sticky;left:0;background:#fafafa}}
 .dur{{color:#999;font-weight:400;margin-left:5px;font-size:11px}}
 .ch{{font-size:10px;letter-spacing:.04em;padding:1px 3px;border-radius:2px;color:#fff}}
 .ch-col{{background:#7a5195}} .ch-lat{{background:#ef5675}} .ch-lon{{background:#3a7ca5}}
 td.bad{{background:#3a3a3a;color:#ffb84d;font-weight:600}}
 .quality{{margin-top:28px;padding:12px 16px;background:#fff4e5;border:1px solid #f0c088;border-radius:5px}}
 .quality li{{margin:5px 0}}
</style></head><body>
<h1>controls biomarkers &mdash; cross-sortie dashboard</h1>
<div class='meta'>{len(summaries)} sorties: {html.escape(sortie_list)}<br>source: {html.escape(src)}</div>
<nav>{nav}</nav>
<p class='meta'>Cells shaded per column: <span style='background:rgb(120,196,120);padding:1px 6px'>green = better</span>
 <span style='background:rgb(214,153,120);padding:1px 6px'>red = worse</span>,
 relative to the 8-sortie range. Neutral metrics (aggressiveness, gain) are unshaded.
 <span style='background:#3a3a3a;color:#ffb84d;padding:1px 6px'>&#9888; = out-of-range / missing</span>.</p>
{tables}
{quality_notes(summaries)}
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true", help="open in browser after writing")
    args = ap.parse_args()
    summaries = load_summaries()
    if not summaries:
        raise SystemExit(f"no _summary.json files under {CONTROLS_DIR}")
    OUT.write_text(render(summaries))
    print(f"Wrote {OUT}  ({len(summaries)} sorties)")
    if args.open:
        webbrowser.open(OUT.as_uri())


if __name__ == "__main__":
    main()
