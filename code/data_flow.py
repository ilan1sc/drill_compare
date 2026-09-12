#!/usr/bin/env python3
"""Orchestrate the check-ride data pipeline end to end, per sortie.

Stages, run in this order (select a subset with --stages):

  pfd_sync           OCR the PFD video -> heading CSV, cross-correlate against
                      telemetry heading for a time offset + drift.
                      (code/pfd.py + code/pfd_compare_csv3.py)
  segmentation        split each sortie into drill attempts/segments.
                      NOT IMPLEMENTED in this repo -- see
                      documents/Segmentation_DB.md (a segmented database
                      exists, sortie1_segmentation.sqlite, but no generating
                      script is checked in). Always reported as skipped.
  audio_biomarkers    extract cockpit audio -> transcribe -> diarize ->
                      glossary tag -> b01-b15 biomarkers.
                      (code/speech/pipeline_best.py)
  flight_control      9 control-handling biomarkers over telemetry.
                      (code/controls_biomarkers/run_all.py)

Discovers, per sortie, its telemetry CSV under --csv-root and its PFD video /
cockpit audio under --video-root (Milestone Smart-Client exports, already
pulled out to plain files under <video-root>/extracted4/<session>/).

    python3 code/data_flow.py --dry-run
    python3 code/data_flow.py --stages flight_control segmentation
    python3 code/data_flow.py --stages pfd_sync --sorties 004 --debug-rois
    python3 code/data_flow.py --stages pfd_sync --sorties 004 --force-pfd-ocr
    python3 code/data_flow.py --stages audio_biomarkers --sorties 004
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEF_CSV_ROOT = Path("/home/ilan/Documents/flight school/raw video/raw_data2/CSV_3.6.26_check1")
DEF_VIDEO_ROOT = Path("/home/ilan/Documents/flight school/raw video/raw_data2/audio_3.6.2026_check1")
STATUS_DIR = ROOT / "exports" / "data_flow"
AUDIO_SRC_DIR = STATUS_DIR / "_audio_src"

STAGES = ["pfd_sync", "segmentation", "audio_biomarkers", "flight_control"]

# pfd.py's built-in ROIs are tuned to a different, 1920x1080 PFD skin and
# land on blank sky/ground on this dataset's 1024x768 clips (verified
# 2026-09-11). code/pfd_rois_check1_1024x768.json replaces them with boxes
# hand-marked and OCR-verified against a real frame (heading/ra/pi/altitude/
# airspeed all read correctly, 2026-09-11/12) -- pass --rois to point at a
# different override, or --force-pfd-ocr to run with pfd.py's defaults anyway.
DEF_PFD_ROIS = ROOT / "code" / "pfd_rois_check1_1024x768.json"


def discover_csv(csv_root: Path, sortie: str) -> Path | None:
    d = csv_root / sortie
    for name in ("Elements_0.csv", "Elements.csv"):
        p = d / name
        if p.exists():
            return p
    return None


def discover_session_dir(video_root: Path, sortie: str) -> Path | None:
    extracted = video_root / "extracted4"
    if not extracted.is_dir():
        return None
    matches = sorted(p for p in extracted.iterdir()
                      if p.is_dir() and p.name.startswith(f"{sortie}_check"))
    return matches[0] if matches else None


def discover_media(session_dir: Path | None, pattern: str) -> Path | None:
    if session_dir is None:
        return None
    hit = next(session_dir.glob(pattern), None)
    if hit:
        return hit
    nested = session_dir / session_dir.name
    if nested.is_dir():
        return next(nested.glob(pattern), None)
    return None


def sorties_in(csv_root: Path) -> list[str]:
    return sorted(p.name for p in csv_root.iterdir() if p.is_dir() and p.name.isdigit())


def discover(csv_root: Path, video_root: Path, sortie: str) -> dict:
    csv_path = discover_csv(csv_root, sortie)
    session_dir = discover_session_dir(video_root, sortie)
    pfd_video = discover_media(session_dir, "F2-GARMIN_PFD*")
    audio_file = discover_media(session_dir, "F2-AUDIO*")
    return {
        "sortie": sortie,
        "csv": str(csv_path) if csv_path else None,
        "session_dir": str(session_dir) if session_dir else None,
        "pfd_video": str(pfd_video) if pfd_video else None,
        "audio_file": str(audio_file) if audio_file else None,
    }


def run_cmd(cmd: list, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-15:])
    if tail:
        print("   ", tail.replace("\n", "\n    "))
    return r


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_pfd_sync(info: dict, args) -> dict:
    if not info["pfd_video"]:
        return {"status": "skipped", "reason": "no PFD video found for this sortie"}
    if not info["csv"]:
        return {"status": "skipped", "reason": "no telemetry CSV found for this sortie"}

    rois_path = Path(args.rois) if args.rois else DEF_PFD_ROIS
    rois_flag = ["--rois", str(rois_path)] if (rois_path and rois_path.exists()) else []

    if args.debug_rois:
        dbg_csv = STATUS_DIR / info["sortie"] / "pfd_roi_check.csv"
        dbg_csv.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["python3", "code/pfd.py", info["pfd_video"], "0", "2",
                  "--debug", *rois_flag, "--out", str(dbg_csv)])
        debug_dir = ROOT / "exports" / "pfd" / "debug"
        crops = {}
        for f in ("heading", "altitude", "ra", "pi", "airspeed"):
            p = debug_dir / f"{f}.png"
            crops[f] = {"path": str(p), "bytes": p.stat().st_size if p.exists() else 0}
        return {"status": "roi_check", "crops": crops, "rois": str(rois_path),
                "note": "inspect exports/pfd/debug/*.png by hand; a near-blank, "
                        "uniform crop under ~300 bytes means the ROI missed the readout"}

    if not rois_flag and not args.force_pfd_ocr:
        return {"status": "blocked",
                "reason": f"no verified ROI file at {rois_path} and pfd.py's built-in "
                          "ROIs are tuned to a different (1920x1080) PFD layout than "
                          "this 1024x768 dataset uses. Pass --rois <file> once one is "
                          "verified (see --debug-rois), or --force-pfd-ocr to run with "
                          "the defaults anyway."}

    # every sortie's PFD video shares the same filename (F2-GARMIN_PFD COCKPIT.mp4),
    # differing only by parent folder -- the sortie id must be in the output name
    # or concurrent/sequential runs across sorties silently clobber each other.
    out_csv = ROOT / "exports" / "pfd" / f"{info['sortie']}_{Path(info['pfd_video']).stem}_pfd.csv"
    t0 = time.time()
    r = run_cmd(["python3", "code/pfd.py", info["pfd_video"], *rois_flag, "--out", str(out_csv)])
    if r.returncode != 0:
        return {"status": "error", "reason": "pfd.py OCR failed", "stderr": r.stderr[-1000:]}
    compare_out = STATUS_DIR / info["sortie"] / "pfd_sync.html"
    r2 = run_cmd(["python3", "code/pfd_compare_csv3.py", "--pfd", str(out_csv),
                   "--data", info["csv"], "--out", str(compare_out)])
    return {"status": "ok" if r2.returncode == 0 else "error",
            "pfd_csv": str(out_csv), "compare_html": str(compare_out),
            "seconds": round(time.time() - t0, 1)}


def stage_segmentation(info: dict, args) -> dict:
    return {"status": "not_implemented",
            "reason": "no segmentation-generating script is checked into this "
                      "repo; sortie1_segmentation.sqlite exists as prior art "
                      "(see documents/Segmentation_DB.md) but isn't callable "
                      "from here yet."}


def stage_audio_biomarkers(info: dict, args) -> dict:
    if not info["audio_file"]:
        return {"status": "skipped", "reason": "no cockpit audio found for this sortie"}
    AUDIO_SRC_DIR.mkdir(parents=True, exist_ok=True)
    named = AUDIO_SRC_DIR / f"{info['sortie']}.m4a"
    if named.exists() or named.is_symlink():
        named.unlink()
    named.symlink_to(info["audio_file"])
    t0 = time.time()
    r = run_cmd(["python3", "code/speech/pipeline_best.py", str(named), "--speakers", "2"])
    return {"status": "ok" if r.returncode == 0 else "error",
            "out_dir": str(ROOT / "exports" / "audio_best" / info["sortie"]),
            "seconds": round(time.time() - t0, 1),
            "stderr_tail": r.stderr[-1000:] if r.returncode != 0 else None}


def stage_flight_control(info: dict, args) -> dict:
    if not info["csv"]:
        return {"status": "skipped", "reason": "no telemetry CSV found for this sortie"}
    t0 = time.time()
    r = run_cmd(["python3", "code/controls_biomarkers/run_all.py", info["csv"],
                  "--sortie", info["sortie"]])
    return {"status": "ok" if r.returncode == 0 else "error",
            "out_dir": str(ROOT / "exports" / "controls" / info["sortie"]),
            "seconds": round(time.time() - t0, 1)}


STAGE_FUNCS = {
    "pfd_sync": stage_pfd_sync,
    "segmentation": stage_segmentation,
    "audio_biomarkers": stage_audio_biomarkers,
    "flight_control": stage_flight_control,
}


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=str(DEF_CSV_ROOT))
    ap.add_argument("--video-root", default=str(DEF_VIDEO_ROOT))
    ap.add_argument("--sorties", nargs="*", default=None,
                     help="sortie ids, e.g. 002 004 008 (default: every numeric "
                          "folder under --csv-root)")
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    ap.add_argument("--dry-run", action="store_true",
                     help="only discover inputs per sortie/stage, run nothing")
    ap.add_argument("--debug-rois", action="store_true",
                     help="pfd_sync: dump ROI crops for a 2s clip instead of OCR-ing the whole video")
    ap.add_argument("--force-pfd-ocr", action="store_true",
                     help="pfd_sync: run full OCR despite the known ROI mismatch")
    ap.add_argument("--rois", default=None,
                     help=f"pfd_sync: --rois override for pfd.py (default: {DEF_PFD_ROIS})")
    args = ap.parse_args()

    csv_root = Path(args.csv_root)
    video_root = Path(args.video_root)
    sorties = args.sorties or sorties_in(csv_root)

    print(f"sorties: {', '.join(sorties)}")
    print(f"stages:  {', '.join(args.stages)}")
    print()

    summary = {}
    for sortie in sorties:
        info = discover(csv_root, video_root, sortie)
        print(f"=== sortie {sortie} ===")
        print(f"  csv:        {info['csv'] or '(none)'}")
        print(f"  pfd_video:  {info['pfd_video'] or '(none)'}")
        print(f"  audio_file: {info['audio_file'] or '(none)'}")

        stage_results = {}
        if not args.dry_run:
            for stage in args.stages:
                print(f"  -- {stage} --")
                result = STAGE_FUNCS[stage](info, args)
                stage_results[stage] = result
                print(f"     status: {result['status']}"
                      + (f"  ({result['reason']})" if result.get("reason") else ""))

        out_dir = STATUS_DIR / sortie
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "status.json").write_text(json.dumps(
            {"sortie": sortie, "discovery": info, "stages": stage_results}, indent=2))
        summary[sortie] = {"discovery": info, "stages": stage_results}
        print()

    (STATUS_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {STATUS_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
