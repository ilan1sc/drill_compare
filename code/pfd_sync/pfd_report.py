#!/usr/bin/env python3
"""Build the HTML report for a pfd_csv_process.py export directory.

Reads --export-root/_run_manifest.json (written by pfd_csv_process.py) and
every cadet's summary.json, and writes one self-contained HTML file:

  - Errors & missing data, up front (sorties with no video, video sessions
    with no telemetry, any cadet that threw during processing)
  - Summary table of every successfully-synced cadet
  - Per-cadet section: sync fit, cleaner stats, exported file paths, and
    the 3 graphs (heading / airspeed / altitude), embedded inline

Independent of pfd_csv_process.py on purpose -- re-run this alone to
rebuild the report after hand-editing a summary.json, without re-running
OCR/clean/sync.

    python3 code/pfd_sync/pfd_report.py --export-root ".../csv_sync"
    python3 code/pfd_sync/pfd_report.py --export-root ".../csv_sync" --out report.html
"""
import argparse
import base64
import html
import json
from datetime import datetime
from pathlib import Path

VERDICT_COLOR = {"CONFIRMED": "#2f7d5d", "LIKELY OK": "#b9770e", "NEEDS REVIEW": "#b3261e"}


def _b64(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return base64.b64encode(p.read_bytes()).decode("ascii")


def load_results(export_root: Path) -> tuple[dict, list[dict]]:
    manifest_path = export_root / "_run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        results = manifest.get("results", [])
    else:
        manifest = {}
        results = []
    # also pick up any summary.json not listed in the manifest (e.g. a
    # manually re-run cadet), and prefer the on-disk version since it may
    # be newer than what's cached in the manifest
    seen = {r.get("label") for r in results}
    by_label = {r.get("label"): r for r in results}
    for d in sorted(export_root.iterdir()):
        sp = d / "summary.json"
        if d.is_dir() and sp.is_file():
            r = json.loads(sp.read_text())
            by_label[r.get("label", d.name)] = r
            seen.add(r.get("label", d.name))
    return manifest, list(by_label.values())


def build_errors_section(manifest: dict, results: list[dict]) -> str:
    items = []
    for s in manifest.get("missing_video_for_sortie", []):
        items.append(f"<li><b>sortie {s}</b> has telemetry but no matching video session "
                      f"(no <code>*.c{int(s)}.*</code> folder under video-root)</li>")
    for s in manifest.get("missing_telemetry_for_video", []):
        items.append(f"<li><b>sortie {s}</b> has a video session but no telemetry csv "
                      f"under csv-root/{s}</li>")
    for r in results:
        if r.get("status") == "skipped":
            items.append(f"<li><b>{html.escape(r.get('label', r.get('sortie', '?')))}</b> "
                          f"skipped &mdash; {html.escape(r.get('reason', ''))}</li>")
        elif r.get("status") == "error":
            items.append(f"<li><b>{html.escape(r.get('label', r.get('sortie', '?')))}</b> "
                          f"<span style='color:#b3261e'>ERROR</span> &mdash; "
                          f"{html.escape(r.get('reason', ''))}</li>")
    if not items:
        return "<section class='ok-banner'>No errors or missing files found in this run.</section>"
    return (f"<section class='errors'><h2>Errors &amp; missing data ({len(items)})</h2>"
            f"<ul>{''.join(items)}</ul></section>")


def build_summary_table(results: list[dict]) -> str:
    ok = [r for r in results if r.get("status") == "ok"]
    ok.sort(key=lambda r: r.get("sortie") or "")
    rows = "\n".join(
        f"<tr><td>{html.escape(r.get('sortie') or '')}</td>"
        f"<td>{html.escape(r['label'])}</td>"
        f"<td>{r['t_scale']:.4f}</td><td>{r['t_shift']:+.1f}s</td>"
        f"<td>{r['heading_score']:.3f}</td><td>{r['airspeed_r']:.3f}</td>"
        f"<td>{r['altitude_r']:.3f}</td><td>{r.get('seconds', '?')}s</td>"
        f"<td><a href='#{html.escape(r['label'])}' style='color:{VERDICT_COLOR[r['verdict']]}'>"
        f"<b>{r['verdict']}</b></a></td></tr>"
        for r in ok
    )
    return f"""<table class="summary">
<tr><th>sortie</th><th>cadet</th><th>scale</th><th>shift</th><th>heading</th>
    <th>airspeed r</th><th>altitude r</th><th>time</th><th>verdict</th></tr>
{rows}
</table>"""


def build_cadet_section(r: dict) -> str:
    if r.get("status") != "ok":
        return ""
    clean_rows = "\n".join(
        f"<tr><td>{f}</td><td>{c['kept']}/{c['raw']}</td>"
        f"<td>{100*c['kept']/max(c['raw'],1):.1f}%</td></tr>"
        for f, c in r["clean"].items()
    )
    img_tags = []
    for k in ("heading_png", "airspeed_png", "altitude_png"):
        b64 = _b64(r["files"][k])
        if b64:
            img_tags.append(f"<figure><img src='data:image/png;base64,{b64}'>"
                             f"<figcaption>{k.replace('_png','')}</figcaption></figure>")
        else:
            img_tags.append(f"<figure class='missing'>missing: {html.escape(r['files'][k])}</figure>")
    file_rows = "\n".join(
        f"<tr><td>{k}</td><td><code>{html.escape(v)}</code></td></tr>"
        for k, v in r["files"].items()
    )
    return f"""
<section id="{html.escape(r['label'])}">
  <h2>{html.escape(r['label'])} &mdash; sortie {html.escape(r.get('sortie') or '?')}
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
        <tr><td>processing time</td><td>{r.get('seconds', '?')}s</td></tr>
      </table>
      <h3>What the cleaner changed</h3>
      <table class="kv"><tr><th>field</th><th>kept/raw</th><th>%</th></tr>{clean_rows}</table>
      <h3>Exported files</h3>
      <table class="kv">{file_rows}</table>
    </div>
    <div class="imgs">{''.join(img_tags)}</div>
  </div>
</section>"""


def write_html_report(manifest: dict, results: list[dict], out_html: Path):
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_err = sum(1 for r in results if r.get("status") == "error")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>PFD sync report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:0 32px 60px;
       background:#f7f8fa;color:#1a1f26}}
 h1{{padding-top:28px}}
 table{{border-collapse:collapse;margin:10px 0}}
 table.summary{{width:100%}}
 table.summary th,table.summary td{{border:1px solid #ddd;padding:7px 12px;text-align:left}}
 table.summary th{{background:#eef1f4}}
 table.kv td,table.kv th{{border:1px solid #ddd;padding:5px 10px;font-size:13.5px;word-break:break-all}}
 .badge{{color:#fff;padding:2px 10px;border-radius:12px;font-size:13px;margin-left:10px}}
 section{{background:#fff;border:1px solid #e2e5e9;border-radius:8px;padding:18px 22px;margin:22px 0}}
 section.errors{{border-color:#e8b4ae;background:#fdf3f2}}
 section.errors h2{{color:#b3261e;margin-top:0}}
 section.ok-banner{{background:#eef8f1;border-color:#bfe3cc;color:#2f7d5d;padding:14px 22px}}
 .cols{{display:flex;gap:24px;flex-wrap:wrap}}
 .cols>div:first-child{{min-width:340px}}
 .imgs{{display:flex;flex-direction:column;gap:6px}}
 .imgs figure{{margin:0}}
 .imgs img{{max-width:640px;width:100%;border:1px solid #ddd;border-radius:4px}}
 .imgs figcaption{{font-size:12px;color:#666;text-align:center}}
 .imgs figure.missing{{padding:20px;color:#b3261e;border:1px dashed #e8b4ae;font-size:12px}}
 .reason{{color:#444;font-style:italic}}
 .stat{{display:inline-block;margin-right:22px;font-size:14px}}
 .stat b{{font-size:20px;display:block}}
 code{{background:#eef1f4;padding:1px 5px;border-radius:3px}}
</style></head><body>
<h1>PFD sync report</h1>
<p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &middot;
   csv-root <code>{html.escape(manifest.get('csv_root', ''))}</code><br>
   video-root <code>{html.escape(manifest.get('video_root', ''))}</code><br>
   export-root <code>{html.escape(manifest.get('export_root', ''))}</code></p>
<p>
  <span class="stat"><b>{n_ok}</b> confirmed/synced</span>
  <span class="stat"><b>{n_skip}</b> skipped</span>
  <span class="stat"><b>{n_err}</b> errored</span>
</p>
{build_errors_section(manifest, results)}
<h2>Summary</h2>
{build_summary_table(results)}
{''.join(build_cadet_section(r) for r in sorted(results, key=lambda r: r.get('sortie') or ''))}
</body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(body)
    print(f"Wrote {out_html}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    export_root = Path(args.export_root)
    manifest, results = load_results(export_root)
    out = Path(args.out) if args.out else export_root / "pfd_report.html"
    write_html_report(manifest, results, out)


if __name__ == "__main__":
    main()
