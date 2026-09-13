#!/usr/bin/env python3
"""End-to-end PFD-CSV pipeline: clean -> sync -> verify -> report, per sortie.

Runs, for every sortie with an OCR'd PFD csv under exports/pfd/:

  1. clean   data_pfd_cleaner.py       -- physically-implausible-jump removal
  2. sync    pfd_compare_csv5.py       -- heading offset + stretch fit
  3. verify  pfd_verify_airspeed.py    -- independent check, same fit
  4. verify  pfd_verify_msl_altitude.py -- independent check, same fit

...then writes:
  - exports/pfd_report/<sortie>/*.png       (3 graphs per sortie)
  - exports/pfd_report/_summary.csv         (one row per sortie: what changed,
                                              fit numbers, recommendation)
  - exports/pfd_report/report.html          (single self-contained report:
                                              summary table + per-sortie
                                              graphs + recommendation)

    python3 code/pfd_sync/pfd_csv_process.py
    python3 code/pfd_sync/pfd_csv_process.py --sorties 004 008
"""
import argparse
import base64
import csv
import html
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_pfd_cleaner                    # noqa: E402
import pfd_compare_csv5 as csv5             # noqa: E402
import pfd_verify_airspeed as vias          # noqa: E402
import pfd_verify_msl_altitude as valt      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
PFD_DIR = ROOT / "exports" / "pfd"
CSV_ROOT = Path("/home/ilan/Documents/flight school/ver_data/26.1/26.1.s1/26.1.s1.csv_3.6.26")
REPORT_DIR = ROOT / "exports" / "pfd_report"

CADETS = {
    "002": "ilan_oria", "003": "rothman_rotkof", "004": "ilan_x",
    "005": "oren_efrati", "006": "sorek_bigeyes", "008": "sorek_lifman",
}

PFD_STEM_RE = re.compile(r"^(\d{3})_.*_pfd$")


def discover_sorties() -> list[str]:
    out = []
    for p in sorted(PFD_DIR.glob("*_pfd.csv")):
        m = PFD_STEM_RE.match(p.stem)
        if m:
            out.append(m.group(1))
    return out


def find_telemetry_csv(sortie: str) -> Path | None:
    d = CSV_ROOT / sortie
    for name in ("Elements_0.csv", "Elements.csv"):
        if (d / name).is_file():
            return d / name
    return None


def find_raw_pfd_csv(sortie: str) -> Path:
    hits = [p for p in PFD_DIR.glob(f"{sortie}_*_pfd.csv")]
    return hits[0]


def recommend(heading_score: float, ias_r: float, alt_r: float) -> tuple[str, str]:
    scores = [x for x in (heading_score, ias_r, alt_r) if x is not None]
    strong = sum(1 for x in scores if x >= 0.85)
    if heading_score is not None and heading_score >= 0.9 and strong >= 3:
        return ("CONFIRMED", "All 3 channels agree -- safe to use this sync for downstream work.")
    if strong >= 2:
        return ("LIKELY OK", "Majority of channels agree; spot-check the odd one out before relying on it.")
    return ("NEEDS REVIEW", "Channels disagree -- do not trust this sync without a closer look "
                            "(narrower scale search, or check for a real video/telemetry duration mismatch).")


def render_heading(pt, ph, dt, dh, res, label, out_png):
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
    fig.suptitle(f"Heading sync -- {label}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def process_sortie(sortie: str) -> dict:
    print(f"=== sortie {sortie} ===")
    cadet = CADETS.get(sortie, sortie)
    raw_pfd = find_raw_pfd_csv(sortie)
    telemetry = find_telemetry_csv(sortie)
    if telemetry is None:
        print("  no telemetry csv found, skipping")
        return {"sortie": sortie, "cadet": cadet, "status": "skipped", "reason": "no telemetry csv"}

    out_dir = REPORT_DIR / sortie
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. clean
    clean_pfd = raw_pfd.with_name(raw_pfd.stem + "_clean.csv")
    clean_report = data_pfd_cleaner.clean_file(raw_pfd, clean_pfd)

    # 2. sync (heading, on the cleaned file)
    pt, ph = csv5.load_pfd(clean_pfd)
    dt, dh = csv5.load_data(telemetry)
    res_hdg = csv5.fit(pt, ph, dt, dh, 0.85, 1.20)
    render_heading(pt, ph, dt, dh, res_hdg, f"{sortie} {cadet}", out_dir / "heading.png")

    # 3. verify: airspeed
    pt_i, pv_i = vias.load_pfd_ias(clean_pfd)
    dt_i, dv_i = vias.load_data_tas(telemetry)
    res_ias = vias.fit_at(pt_i, pv_i, dt_i, dv_i, res_hdg["t_scale"], res_hdg["t_shift"])
    vias.render(pt_i, pv_i, dt_i, dv_i, res_ias, f"{sortie} {cadet}", out_dir / "airspeed.png")

    # 4. verify: altitude
    pt_a, pv_a = valt.load_pfd_alt(clean_pfd)
    dt_a, dv_a = valt.load_data_msl(telemetry)
    res_alt = valt.fit_at(pt_a, pv_a, dt_a, dv_a, res_hdg["t_scale"], res_hdg["t_shift"])
    valt.render(pt_a, pv_a, dt_a, dv_a, res_alt, f"{sortie} {cadet}", out_dir / "altitude.png")

    verdict, reason = recommend(res_hdg["score"], res_ias["r"], res_alt["r"])
    print(f"  heading mean-cos {res_hdg['score']:.4f}   airspeed r {res_ias['r']:.3f}   "
          f"altitude r {res_alt['r']:.3f}   -> {verdict}")

    return {
        "sortie": sortie, "cadet": cadet, "status": "ok",
        "t_scale": res_hdg["t_scale"], "t_shift": res_hdg["t_shift"],
        "heading_score": res_hdg["score"], "heading_rms_deg": res_hdg["rms_deg"],
        "airspeed_r": res_ias["r"], "altitude_r": res_alt["r"],
        "clean": clean_report, "verdict": verdict, "verdict_reason": reason,
        "graphs": {"heading": str(out_dir / "heading.png"),
                   "airspeed": str(out_dir / "airspeed.png"),
                   "altitude": str(out_dir / "altitude.png")},
    }


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def write_summary_csv(results: list[dict], out_csv: Path):
    fields = ["sortie", "cadet", "status", "t_scale", "t_shift", "heading_score",
              "heading_rms_deg", "airspeed_r", "altitude_r", "verdict", "verdict_reason",
              "airspeed_kept_pct", "altitude_kept_pct", "heading_kept_pct",
              "ra_kept_pct", "pi_kept_pct"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            if r.get("status") == "ok":
                for f in ("airspeed", "altitude", "heading", "ra", "pi"):
                    c = r["clean"].get(f)
                    row[f"{f}_kept_pct"] = round(100 * c["kept"] / max(c["raw"], 1), 1) if c else ""
            w.writerow(row)
    print(f"Wrote {out_csv}")


VERDICT_COLOR = {"CONFIRMED": "#2f7d5d", "LIKELY OK": "#b9770e", "NEEDS REVIEW": "#b3261e"}


def write_html_report(results: list[dict], out_html: Path):
    ok = [r for r in results if r.get("status") == "ok"]
    rows = "\n".join(
        f"<tr><td>{r['sortie']}</td><td>{html.escape(r['cadet'])}</td>"
        f"<td>{r['t_scale']:.4f}</td><td>{r['t_shift']:+.1f}s</td>"
        f"<td>{r['heading_score']:.3f}</td><td>{r['airspeed_r']:.3f}</td>"
        f"<td>{r['altitude_r']:.3f}</td>"
        f"<td><a href='#s{r['sortie']}' style='color:{VERDICT_COLOR[r['verdict']]}'>"
        f"<b>{r['verdict']}</b></a></td></tr>"
        for r in ok
    )

    sections = []
    for r in ok:
        clean_rows = "\n".join(
            f"<tr><td>{f}</td><td>{c['kept']}/{c['raw']}</td>"
            f"<td>{100*c['kept']/max(c['raw'],1):.1f}%</td></tr>"
            for f, c in r["clean"].items()
        )
        imgs = "".join(
            f"<figure><img src='data:image/png;base64,{_b64(r['graphs'][k])}'>"
            f"<figcaption>{k}</figcaption></figure>"
            for k in ("heading", "airspeed", "altitude")
        )
        sections.append(f"""
<section id="s{r['sortie']}">
  <h2>Sortie {r['sortie']} &mdash; {html.escape(r['cadet'])}
    <span class="badge" style="background:{VERDICT_COLOR[r['verdict']]}">{r['verdict']}</span></h2>
  <p class="reason">{html.escape(r['verdict_reason'])}</p>
  <div class="cols">
    <div>
      <h3>Sync fit</h3>
      <table class="kv">
        <tr><td>time model</td><td>data_t = {r['t_scale']:.4f} &middot; pfd_t {r['t_shift']:+.1f}s</td></tr>
        <tr><td>heading match</td><td>mean-cos {r['heading_score']:.4f} (~{r['heading_rms_deg']:.1f}&deg; err)</td></tr>
        <tr><td>airspeed verify</td><td>r = {r['airspeed_r']:.3f}</td></tr>
        <tr><td>altitude verify</td><td>r = {r['altitude_r']:.3f}</td></tr>
      </table>
      <h3>What the cleaner changed</h3>
      <table class="kv"><tr><th>field</th><th>kept/raw</th><th>%</th></tr>{clean_rows}</table>
    </div>
    <div class="imgs">{imgs}</div>
  </div>
</section>""")

    skipped = [r for r in results if r.get("status") != "ok"]
    skipped_html = ""
    if skipped:
        items = "".join(f"<li>{r['sortie']}: {html.escape(r.get('reason',''))}</li>" for r in skipped)
        skipped_html = f"<p><b>Skipped:</b></p><ul>{items}</ul>"

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>PFD CSV process report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:0 32px 60px;
       background:#f7f8fa;color:#1a1f26}}
 h1{{padding-top:28px}}
 table{{border-collapse:collapse;margin:10px 0}}
 table.summary{{width:100%}}
 table.summary th,table.summary td{{border:1px solid #ddd;padding:7px 12px;text-align:left}}
 table.summary th{{background:#eef1f4}}
 table.kv td,table.kv th{{border:1px solid #ddd;padding:5px 10px;font-size:13.5px}}
 .badge{{color:#fff;padding:2px 10px;border-radius:12px;font-size:13px;margin-left:10px}}
 section{{background:#fff;border:1px solid #e2e5e9;border-radius:8px;padding:18px 22px;margin:22px 0}}
 .cols{{display:flex;gap:24px;flex-wrap:wrap}}
 .cols>div:first-child{{min-width:320px}}
 .imgs{{display:flex;flex-direction:column;gap:6px}}
 .imgs figure{{margin:0}}
 .imgs img{{max-width:640px;width:100%;border:1px solid #ddd;border-radius:4px}}
 .imgs figcaption{{font-size:12px;color:#666;text-align:center}}
 .reason{{color:#444;font-style:italic}}
 code{{background:#eef1f4;padding:1px 5px;border-radius:3px}}
</style></head><body>
<h1>PFD CSV process report</h1>
<p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &middot;
   pipeline: <code>data_pfd_cleaner.py</code> &rarr; <code>pfd_compare_csv5.py</code> &rarr;
   <code>pfd_verify_airspeed.py</code> + <code>pfd_verify_msl_altitude.py</code></p>

<table class="summary">
<tr><th>sortie</th><th>cadet</th><th>scale</th><th>shift</th><th>heading</th>
    <th>airspeed r</th><th>altitude r</th><th>verdict</th></tr>
{rows}
</table>
{skipped_html}
{''.join(sections)}
</body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc)
    print(f"Wrote {out_html}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sorties", nargs="*", default=None)
    args = ap.parse_args()

    sorties = args.sorties or discover_sorties()
    print(f"sorties: {', '.join(sorties)}\n")

    results = [process_sortie(s) for s in sorties]
    write_summary_csv(results, REPORT_DIR / "_summary.csv")
    write_html_report(results, REPORT_DIR / "report.html")


if __name__ == "__main__":
    main()
