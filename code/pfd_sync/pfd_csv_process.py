#!/usr/bin/env python3
"""End-to-end PFD pipeline: OCR -> clean -> sync -> verify -> export, per cadet.

Runs, for every cadet session found under --video-root that has a matching
telemetry folder under --csv-root:

  0. OCR      pfd2.py --preset fast     -- read the PFD video (skipped if a
                                            raw OCR csv already exists there);
                                            pass --legacy-ocr for pfd.py instead
  1. clean    data_pfd_cleaner.py       -- physically-implausible-jump removal
  2. sync     pfd_compare_csv5.py       -- heading offset + stretch fit
  3. verify   pfd_verify_airspeed.py    -- independent check, same fit
  4. verify   pfd_verify_msl_altitude.py -- independent check, same fit

...then writes, per cadet, under --export-root/<cadet library name>/:
  - pfd_raw.csv            OCR output, unmodified
  - pfd_clean.csv          after data_pfd_cleaner
  - telemetry_synced.csv   the telemetry csv with PFD-clock columns inserted
  - heading.png / airspeed.png / altitude.png
  - summary.json           every number above, in one place

...and, once every cadet is done, at --export-root:
  - _summary.csv           one row per cadet (this run's "actions" record)
  - _run_manifest.json     what was found, skipped, or errored, and why
                           (pfd_report.py turns this + the summaries into the
                           actual HTML report -- this script only produces
                           the per-cadet data)

    python3 code/pfd_sync/pfd_csv_process.py \\
        --csv-root ".../26.1.s1.csv_3.6.26" \\
        --video-root ".../26.1.s1.extrct_v2" \\
        --export-root ".../csv_sync"
"""
import argparse
import csv
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_pfd_cleaner                    # noqa: E402
import pfd_compare_csv5 as csv5             # noqa: E402
import pfd_verify_airspeed as vias          # noqa: E402
import pfd_verify_msl_altitude as valt      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
DEF_ROIS = Path(__file__).resolve().parent / "pfd_rois_check1_1024x768.json"

# cadet-library folder name -> sortie number: "...c<N>...." or "...c<N>_exp_v2"
CADET_RE = re.compile(r"\.c(\d+|x)\.")


def discover_cadet_sessions(video_root: Path) -> list[dict]:
    """Every '*.c<N>.<name>[_exp_vN]' folder under video_root that has a
    PFD video in it (directly, or one level down -- extraction layouts
    differ)."""
    out = []
    for d in sorted(video_root.iterdir()):
        if not d.is_dir():
            continue
        m = CADET_RE.search(d.name)
        if not m:
            continue
        code = m.group(1)
        sortie = None if code == "x" else f"{int(code):03d}"
        pfd = next(d.glob("*F2-GARMIN_PFD*"), None) or next(d.glob("*/*F2-GARMIN_PFD*"), None)
        label = d.name[:-len("_exp_v2")] if d.name.endswith("_exp_v2") else d.name
        out.append({"folder": d, "label": label, "sortie": sortie, "pfd_video": pfd})
    return out


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


def write_synced_telemetry(telemetry: Path, res: dict, out_csv: Path):
    """telemetry csv + two extra leading columns giving each row's time on
    the PFD video's own clock: pfd_t = (data_t - t_shift) / t_scale."""
    raw = pd.read_csv(telemetry, sep=None, engine="python")
    data_s = pd.to_numeric(raw[raw.columns[0]], errors="coerce") / 1000.0
    pfd_s = (data_s - res["t_shift"]) / res["t_scale"]
    aligned = raw.copy()
    aligned.insert(0, "PFD Time (s)", pfd_s.round(4))
    aligned.insert(1, "PFD Time (ms)", (pfd_s * 1000.0).round(1))
    aligned.to_csv(out_csv, index=False)


def process_cadet(session: dict, csv_root: Path, export_root: Path, rois: Path,
                   force_ocr: bool, legacy_ocr: bool = False) -> dict:
    label, sortie = session["label"], session["sortie"]
    print(f"=== {label}  (sortie {sortie}) ===")
    out_dir = export_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if sortie is None:
        print("  no numbered sortie (extra/unmatched session), skipping")
        return {"label": label, "sortie": None, "status": "skipped",
                "reason": "cadet folder has no numbered sortie (e.g. 'cx') -- no telemetry to sync against"}
    if session["pfd_video"] is None:
        print("  no PFD video found in this cadet's folder, skipping")
        return {"label": label, "sortie": sortie, "status": "skipped",
                "reason": "no F2-GARMIN_PFD video found in this cadet's folder"}
    telemetry = find_telemetry_csv(csv_root, sortie)
    if telemetry is None:
        print("  no telemetry csv found, skipping")
        return {"label": label, "sortie": sortie, "status": "skipped",
                "reason": f"no Elements[_0].csv under {csv_root / sortie}"}

    try:
        # 0. OCR (skip if already done and not forcing)
        raw_pfd = out_dir / "pfd_raw.csv"
        if force_ocr or not raw_pfd.exists():
            import subprocess
            ocr_script = "pfd.py" if legacy_ocr else "pfd2.py"
            cmd = ["python3", str(Path(__file__).resolve().parent / ocr_script),
                   str(session["pfd_video"]), "--rois", str(rois), "--out", str(raw_pfd)]
            if not legacy_ocr:
                cmd += ["--preset", "fast"]
            print("  $", " ".join(cmd))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not raw_pfd.exists():
                raise RuntimeError(f"pfd.py OCR failed: {r.stderr[-1500:]}")
        else:
            print(f"  OCR skipped, {raw_pfd} already exists")

        # 1. clean
        clean_pfd = out_dir / "pfd_clean.csv"
        clean_report = data_pfd_cleaner.clean_file(raw_pfd, clean_pfd)

        # 2. sync (heading, on the cleaned file)
        pt, ph = csv5.load_pfd(clean_pfd)
        dt, dh = csv5.load_data(telemetry)
        res_hdg = csv5.fit(pt, ph, dt, dh, 0.85, 1.20)
        render_heading(pt, ph, dt, dh, res_hdg, label, out_dir / "heading.png")

        # 3. verify: airspeed
        pt_i, pv_i = vias.load_pfd_ias(clean_pfd)
        dt_i, dv_i = vias.load_data_tas(telemetry)
        res_ias = vias.fit_at(pt_i, pv_i, dt_i, dv_i, res_hdg["t_scale"], res_hdg["t_shift"])
        vias.render(pt_i, pv_i, dt_i, dv_i, res_ias, label, out_dir / "airspeed.png")

        # 4. verify: altitude
        pt_a, pv_a = valt.load_pfd_alt(clean_pfd)
        dt_a, dv_a = valt.load_data_msl(telemetry)
        res_alt = valt.fit_at(pt_a, pv_a, dt_a, dv_a, res_hdg["t_scale"], res_hdg["t_shift"])
        valt.render(pt_a, pv_a, dt_a, dv_a, res_alt, label, out_dir / "altitude.png")

        # export the synced telemetry csv
        write_synced_telemetry(telemetry, res_hdg, out_dir / "telemetry_synced.csv")

        verdict, reason = recommend(res_hdg["score"], res_ias["r"], res_alt["r"])
        print(f"  heading mean-cos {res_hdg['score']:.4f}   airspeed r {res_ias['r']:.3f}   "
              f"altitude r {res_alt['r']:.3f}   -> {verdict}  ({time.time()-t0:.0f}s)")

        result = {
            "label": label, "sortie": sortie, "status": "ok",
            "telemetry_csv": str(telemetry), "pfd_video": str(session["pfd_video"]),
            "t_scale": res_hdg["t_scale"], "t_shift": res_hdg["t_shift"],
            "heading_score": res_hdg["score"], "heading_rms_deg": res_hdg["rms_deg"],
            "airspeed_r": res_ias["r"], "altitude_r": res_alt["r"],
            "clean": clean_report, "verdict": verdict, "verdict_reason": reason,
            "seconds": round(time.time() - t0, 1),
            "files": {
                "pfd_raw": str(raw_pfd), "pfd_clean": str(clean_pfd),
                "telemetry_synced": str(out_dir / "telemetry_synced.csv"),
                "heading_png": str(out_dir / "heading.png"),
                "airspeed_png": str(out_dir / "airspeed.png"),
                "altitude_png": str(out_dir / "altitude.png"),
            },
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        result = {"label": label, "sortie": sortie, "status": "error",
                  "reason": str(e), "traceback": traceback.format_exc()}

    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def write_summary_csv(results: list[dict], out_csv: Path):
    fields = ["label", "sortie", "status", "t_scale", "t_shift", "heading_score",
              "heading_rms_deg", "airspeed_r", "altitude_r", "verdict", "verdict_reason",
              "seconds", "airspeed_kept_pct", "altitude_kept_pct", "heading_kept_pct",
              "ra_kept_pct", "pi_kept_pct", "reason"]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--export-root", required=True)
    ap.add_argument("--rois", default=str(DEF_ROIS))
    ap.add_argument("--sorties", nargs="*", default=None,
                    help="restrict to these sortie numbers (e.g. 004 008)")
    ap.add_argument("--force-ocr", action="store_true", help="re-run OCR even if pfd_raw.csv exists")
    ap.add_argument("--legacy-ocr", action="store_true",
                    help="use pfd.py (full-quality, ~1.7-3x slower) instead of "
                         "pfd2.py --preset fast (the default since 2026-09-13)")
    args = ap.parse_args()

    csv_root = Path(args.csv_root)
    video_root = Path(args.video_root)
    export_root = Path(args.export_root)
    rois = Path(args.rois)
    export_root.mkdir(parents=True, exist_ok=True)

    sessions = discover_cadet_sessions(video_root)
    if args.sorties:
        wanted = set(args.sorties)
        sessions = [s for s in sessions if s["sortie"] in wanted]
    print(f"found {len(sessions)} cadet session(s) under {video_root}\n")

    csv_sorties = {p.name for p in csv_root.iterdir() if p.is_dir() and p.name.isdigit()}
    video_sorties = {s["sortie"] for s in sessions if s["sortie"]}
    missing_video = sorted(csv_sorties - video_sorties)
    missing_csv_for_video = sorted(s["sortie"] for s in sessions
                                    if s["sortie"] and s["sortie"] not in csv_sorties)

    results = [process_cadet(s, csv_root, export_root, rois, args.force_ocr, args.legacy_ocr)
               for s in sessions]

    write_summary_csv(results, export_root / "_summary.csv")
    manifest = {
        "csv_root": str(csv_root), "video_root": str(video_root), "export_root": str(export_root),
        "csv_sorties": sorted(csv_sorties), "video_sessions": [s["label"] for s in sessions],
        "missing_video_for_sortie": missing_video,
        "missing_telemetry_for_video": missing_csv_for_video,
        "results": results,
    }
    (export_root / "_run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Wrote {export_root / '_run_manifest.json'}")


if __name__ == "__main__":
    main()
