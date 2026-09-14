# agents_claude.md — orientation for a future me (or you)

Written 2026-09-14, at the end of a long session. If you're reading this
because you came back after months away and remember nothing: start
here, then read each area's own README before touching its code.

## What this repo is

Flight-school check-ride data pipeline. Cadets fly a helicopter
simulator; each sortie is recorded as several synchronized camera feeds
+ audio + 20 Hz simulator telemetry (`Elements_0.csv` — heading,
attitude, controls, etc). The overall goal is to turn that raw capture
into aligned, analyzable data: where the cadet looked, how they flew,
what was said, when drills happened — all on one shared clock.

Data lives OUTSIDE this repo, under `/home/ilan/Documents/flight school/`
(videos are large; `*.mp4`/`*.mov` etc are gitignored). Several parallel
dataset generations exist there — the current/canonical one as of this
session is:
```
/home/ilan/Documents/flight school/ver_data/26.1/26.1.s1/
  26.1.s1.csv_3.6.26/                 -- telemetry CSVs, per sortie (001-008)
  26.1.s1.extrct_v2/                  -- per-cadet folders, each with:
    26.1.s1.c<N>.<name>_exp_v2/
      00_F2-AUDIO...m4a
      01_F2-GARMIN_PFD COCKPIT.mp4    -- PFD instrument screen recording
      02_F2-PILOT CAM COCKPIT-...mp4  -- rear/over-the-shoulder cockpit view
      03_F2-IG12...mp4                -- outside view
      04_F2-NADIV_COCKPIT...mp4
      05_F2-GARMIN_MFD COCKPIT.mp4
      06_F2-COPILOT CAM COCKPIT-...mp4 -- face-on camera, sees the crew
      manifest.json                    -- IMPORTANT: has each stream's
                                           wall-clock start_unix -- use
                                           this for cross-camera sync,
                                           don't re-derive it by OCR
      sync_report.json
```
Cadet naming: `26.1.s1.c<N>.<firstname>_<lastname/callsign>`. The
**cadet always sits LEFT seat** in these recordings (confirmed by the
user). Earlier dataset generations (`raw_data`, `raw_data2`,
`raw_video_codex`, etc.) exist too — check with the user before
assuming which one is "current."

## The live subsystems (merged from two concurrent sessions on 2026-09-14)

### 1. `code/pfd_sync/` — PFD-video-to-telemetry time sync

**Read `code/pfd_sync/README.md` first** — it's comprehensive (CLI
syntax, pipeline diagram, function-by-function reference, an OCR-speed
benchmark table, and a documented-but-unfixed ~4.8° magnetic-heading
residual). Short version: OCRs the `01_F2-GARMIN_PFD` video's on-screen
heading/altitude/RA/airspeed, cross-correlates against telemetry to find
`(t_shift, t_scale)`, verifies the fit against 2 independent channels,
exports synced telemetry + an HTML report.

Entry point: `pfd_csv_process.py` (orchestrator) → `pfd_report.py`
(report builder). OCR default is `pfd2.py --preset fast` (~1.5-1.7x
faster than the original `pfd.py`, same accuracy — see README for the
benchmark). All 8 sorties (001-008) were CONFIRMED synced as of this
session, output under
`ver_data/26.1/26.1.s1/csv_sync/` and `csv6sync/` (csv6 = faster
re-sync-only tool, `--reuse-root` mode).

**Not yet done**: fitting out the ~4.8° magnetic/true heading offset as
a 3rd parameter (documented, not implemented — would tighten `rms_deg`
reporting, doesn't affect correctness of the sync itself).
**Known inconsistency**: `code/data_flow.py` (a separate, older
orchestrator, NOT in `pfd_sync/`) still calls the legacy `pfd.py` +
`pfd_compare_csv3.py` directly and was never updated to the new
pipeline. Don't assume it's using the current code.

### 2. `gazelab2/` — gaze-detection accuracy lab (a benchmark, not a pipeline)

One-off investigation: why was `code/gaze/extract_gaze.py`'s face
detection only hitting ~9% coverage on the face-cam video? Answer:
processing the full frame (both seats) instead of cropping to the
cadet's (left) seat. Cropping alone took coverage 24.7% → 97.6%;
upscaling/enhancement variants tested past that were all noise-level.
Bonus finding: the existing gray-world white-balance preprocessing step
in `extract_gaze.py` doesn't actually help on this footage — dropping it
gave the best result of everything tested.

**Read `gazelab2/README.md`** for the full comparison table. This
finding fed directly into `code/cockpit_gaze/` below (which uses the
crop+no-upscale+no-enhance config as its extraction settings) but was
**not** back-ported into `code/gaze/extract_gaze.py` itself — that file
still has the old (worse) defaults. If gaze coverage on the original
`code/gaze/` pipeline ever needs fixing, this is the fix, already proven
out, just not applied there yet.

Needs its own venv (`gazelab2/venv/`, gitignored) — the repo's global
Python env has `protobuf` 7.x, which is incompatible with `mediapipe`
0.10.18's `FaceLandmarker` (breaks with `MessageFactory.GetPrototype`
errors). `gazelab2/venv` pins `protobuf<4.24`. **`code/cockpit_gaze/`
below needs this same venv** — global Python can't run mediapipe here.

### 3. `code/cockpit_gaze/` — gaze projected onto the cockpit view (newest, least mature)

Built in this session, tested on ONE cadet (`26.1.s1.c6`), ONE 2-minute
window. **Read `code/cockpit_gaze/README.md`** for full details and
known limitations before trusting or extending this.

What it does: extracts the cadet's gaze from the face-on camera (`06`,
left seat, using gazelab2's winning crop/no-enhance config), clusters it
into on_pfd / other_inside / outside (reusing
`code/atm/atm_gaze_cluster.py`'s auto-calibration logic), finds the PFD
screen's pixel box in the cockpit-view camera (`02`) via
`pfd_finder.py`, and projects the gaze angle onto that camera's pixel
plane, pinned at the PFD. Outputs: two augmented debug videos, a
per-frame CSV (t, x, y, state, yaw, pitch), a static heatmap, and a
`calibration.json` (the anchor+scale "homography" — NOT a true planar
homography, see the README for why).

**Correction made mid-session**: the PFD is the **RIGHT-hand** of the
two console screens in camera `02` (map/nav display), not the left one
(round-dial six-pack) — confirmed against a user-annotated frame
(green=PFD, red=not-PFD). `pfd_finder.py` was fixed to pick the
rightmost bright screen; if you see it picking the wrong one again,
check that logic first (`code/cockpit_gaze/pfd_finder.py`, the
`big.sort(...)` + `big[-1]` line).

**Explicitly unvalidated, don't trust past a demo**:
- `--hfov`/`--vfov` (90°/70°) are guessed, not measured from the actual
  lens on camera `02`.
- `flip_x=True` (yaw sign flips between camera `06` and camera `02`,
  because the mounts face opposite directions) is a physical-plausibility
  assumption, not confirmed against a known ground-truth glance.
- Only run on `26.1.s1.c6`, only the first 2 minutes. The other 7
  cadets and the rest of each sortie are completely untested.
- The `02` camera view showing the cockpit-forward/PFD area was
  originally misidentified (I initially probed the wrong file,
  `02_F2-PILOT CAM COCKPIT`, thinking it was a rear-view-of-crew-backs
  shot with no faces — that's correct, `02` has no faces, it's the
  cockpit/instrument view; the face-on camera is `06`. Don't confuse
  these two again.)

### 4. `code/speech/`, `code/speech_v2/` — ASR/diarization/CRM-biomarker pipeline on cockpit audio

Not touched by the session that wrote the rest of this file (that session
found `code/speech/process_audio_v1.py` already staged in git and included
it in the sync commit as-is) — this section is from the session that
actually built it, so it's the authoritative account. Read
`code/speech/audio_res.md` for the full "why" (engine bake-off results).

**Winning engines** (bake-off run 2026-09-05, still current):
ASR = `ivrit-ai/whisper-large-v3` (Hebrew-finetuned Whisper, via
`transcribe_ivrit_v3.py` — best coverage, least hallucination/looping of
6 backends tried). Diarization = 3D-Speaker/CAM++ (Alibaba, via
ModelScope, `diarize_3dspeaker.py` — cleanest 2-speaker split, most
consistent across sorties, can self-pick speaker count with
`--speakers auto`). Plus `glossary.py` (aviation term tagging) and
`code/biomarkers/b01..b15_*.py` (CRM/communication biomarkers — observation/
intention/action/verification statements, OIAV loop completion, error
correction, hesitations, disfluencies, comm-relevance, etc), run via
`run_biomarkers.py`.

**Run it with `code/speech/process_audio_v1.py`** — packages trim → transcribe →
diarize → glossary-tag → biomarkers → optional augmented visualization movie
(speaker-color banner + RTL Hebrew subtitles + live biomarker panel that
lights up as each b01-b15 fires, via `audio_vis_biomarkers.py` + `av_sync.py`)
into one command:
```bash
# whole sortie (no --duration => probes file length, runs full length)
python3 code/speech/process_audio_v1.py \
    "<cadet source dir, e.g. .../26.1.s1.c6.sorek_bigeyes_exp_v2>" \
    --out-dir "<cadet library dir, e.g. .../26.1.s1.audio/26.1.s1.c6.sorek_bigeyes>"

# short sample, or --skip-movie for transcript+biomarkers only (much faster)
python3 code/speech/process_audio_v1.py "<cadet source dir>" \
    --out-dir "<cadet library dir>" --start 600 --duration 60
```
First arg can be an exact video file or a cadet source dir (auto-picks
`06_F2-COPILOT CAM COCKPIT*`, override with `--cam-glob`); `--audio-source`
swaps in the dedicated mic feed (`00_F2-AUDIO(...).m4a`) instead of the
video's own track. The trimmed video used to render the movie is a temp
file, deleted after use — the output library never keeps a video copy.
Full-length movie rendering is slow (roughly on the order of the video's
own length) — a 7-cadet full-sortie batch (`--skip-movie`, transcript+
diarize+biomarkers only) ran 3-way parallel on this machine's GPU (NVIDIA
GB10, 124GB unified memory — plenty of headroom, CPU-bound ffmpeg/render
is the actual bottleneck) and finished in a few hours.

**Output library**: `/home/ilan/Documents/flight school/ver_data/26.1/26.1.s1/26.1.s1.audio/26.1.s1.c<N>.<name>/sample_<name>/`
— `transcript.json`, `glossary_review_queue.csv`, `biomarkers/` (b01-b15 +
`_summary.json`), `.wav`/`.16k.wav` (not auto-deleted), and
`biomarker_vis_<start>_<dur>.mp4` + `.sync.json` when the movie wasn't
skipped. Mirrors the layout already established in `exports/audio_best/`
(gitignored, from the original 2026-09-06/07 single-sortie work).

As of 2026-09-14: full-sortie batch done for c1/c2/c3/c4/c5/c7/c8
(**c6 explicitly skipped** — only has short samples, `sample_1min`/
`sample_60s`, from earlier testing). Quick read on the results (obs/min,
act/min, comm-relevance share — see each cadet's `biomarkers/_summary.json`
for the full breakdown): c1=2.17/2.11/0.53, c2=0.40/0.11/0.29,
c3=3.35/0.91/0.43, c4=4.21/2.28/0.58, c5=2.24/1.29/0.46, c7=3.09/1.49/0.47,
c8=0.54/0.10/0.27. **c2 and c8 are notable low outliers** (~5-10x lower
obs/act rates, comm-relevance ~0.27-0.29 vs ~0.43-0.58 elsewhere) —
`audio_res.md` flagged a similar pattern for c2 back in the original
bake-off too, so it may be a real, reproducible property of that cadet's
recording rather than a pipeline bug, but it hasn't been root-caused.
Full-length `biomarker_vis` movies for all 7 were queued to render right
after the transcript/biomarker batch — check each cadet's sample dir for
whether that finished, and `26.1.s1.audio/_run_logs/` for the logs.

- `code/gaze/` — the original gaze biomarker suite (AOI dwell time,
  scan-path regularity, fixation duration, etc.) built on top of
  `extract_gaze.py`. Has the coverage bug `gazelab2` found and fixed the
  recipe for, but the fix wasn't applied here.
- `code/atm/` — ATM-camera gaze pairing/clustering/visualization
  (`atm_gaze_cluster.py`, `atm_gaze_pair.py`, `atm_gaze_vis.py`,
  `sync_ocr.py`). `cockpit_gaze` reuses `atm_gaze_cluster.py`'s
  clustering logic directly (imported, not copied).
- `code/controls_biomarkers/` — flight-control-handling biomarkers
  (precision, smoothness, overshoot, etc) from telemetry alone, plus a
  cross-sortie dashboard.
- A segmentation database, `sortie1_segmentation.sqlite` (drill
  attempt/segment boundaries per sortie) exists and is committed, but
  per `code/pfd_sync/README.md` and this repo's history, it is **not
  wired into** any of video/audio/flight-control processing yet — every
  one of those still runs over whole sorties, not per-segment.
- `checkride_pipeline.pptx` (in `~/Downloads/`, outside the repo) — a
  slide deck describing the overall pipeline, last updated this session
  to reflect the `pfd2`/`csv6` speedup and 3-channel verification. Built
  via a `python-pptx` script that lived in a since-recycled scratchpad
  venv — if you need to rebuild it, there's no committed source for the
  deck-builder script; recreate from scratch referencing the README docs.

## Environment gotchas

- **mediapipe needs `protobuf<4.24`**; the global env has 7.x. Any
  mediapipe-based work (gaze extraction) needs `gazelab2/venv/bin/python`,
  not plain `python3`. If that venv is gone, recreate it:
  `python3 -m venv gazelab2/venv && gazelab2/venv/bin/pip install mediapipe opencv-python numpy "protobuf<4.24"`
- No GPU acceleration available for onnxruntime (OCR) on this machine
  (GB10 / aarch64 Grace-Blackwell) — only CPU/Azure providers. Don't
  waste time chasing a CUDA onnxruntime install here.
- `rapidocr_onnxruntime` is the OCR engine used throughout `pfd_sync`.

## Git

Remote: `origin` = `https://github.com/ilan1sc/drill_compare.git`,
branch `main`. Only commit/push when explicitly asked — this has been
the norm all session. `.gitignore` excludes `exports/`, `venv/`/`.venv/`,
video/audio media, `__pycache__`.

## If you only read one thing

Read the README inside whichever of `code/pfd_sync/`, `gazelab2/`, or
`code/cockpit_gaze/` you're about to touch, or `code/speech/audio_res.md`
for the speech pipeline — each is kept current and has more detail (CLI
syntax, exact numbers, exact limitations) than this file. This file is
the map; those docs are the territory.
