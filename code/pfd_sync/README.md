# pfd_sync

Turns a Garmin PFD cockpit video into a verified time-sync between that
video's clock and the simulator's 20 Hz telemetry, then exports the
synced data and a report — for one cadet or a whole check-ride's worth.

## Why this exists

The PFD video and the telemetry CSV are recorded on two different
clocks, offset by a few seconds to a few minutes and (rarely) drifting
at a slightly different rate. Reading the PFD's on-screen instruments
via OCR gives a second copy of heading/airspeed/altitude/radar-altitude/
torque — the same physical quantities the telemetry already has — so
cross-correlating the two finds the time relationship between them.
Heading is the primary signal (it sweeps the full 0-360° range during
pattern work, so it's the least ambiguous); airspeed and altitude are
used afterwards purely to *verify* that fit from two unrelated channels,
not to find it.

---

## CLI — how to run it

### 1. The whole pipeline, for every cadet found

```bash
python3 code/pfd_sync/pfd_csv_process.py \
    --csv-root    ".../26.1.s1.csv_3.6.26" \
    --video-root  ".../26.1.s1.extrct_v2" \
    --export-root ".../csv_sync"
```

| flag | meaning |
|---|---|
| `--csv-root` | folder containing `<sortie>/Elements_0.csv` (or `Elements.csv`) per sortie |
| `--video-root` | folder containing `*.c<N>.<cadet name>[_exp_vN]/` session folders, each with a PFD video |
| `--export-root` | where per-cadet output folders get written |
| `--sorties 004 008` | optional: restrict to these sortie numbers instead of every one found |
| `--rois <file>` | optional: ROI override (default `pfd_rois_check1_1024x768.json`, next to this script) |
| `--force-ocr` | re-run OCR even if that cadet's `pfd_raw.csv` already exists |
| `--legacy-ocr` | use `pfd.py` (full-quality, ~1.5-1.7x slower) instead of the default `pfd2.py --preset fast` |

### 2. Build (or rebuild) the HTML report

```bash
python3 code/pfd_sync/pfd_report.py --export-root ".../csv_sync"
```

Independent of step 1 on purpose — it only reads what step 1 already
wrote (`_run_manifest.json` + every `summary.json`), so re-run it alone
any time without redoing OCR/sync/verify (e.g. after hand-editing a
`summary.json`).

### 3. Fast re-sync of already-OCR'd data (no OCR at all)

If a cadet's `pfd_clean.csv` already exists somewhere (from step 1, or
from a previous run), re-fit it in seconds instead of minutes:

```bash
# every cadet under an existing export dir
python3 code/pfd_sync/pfd_compare_csv6.py \
    --csv-root    ".../26.1.s1.csv_3.6.26" \
    --reuse-root  ".../csv_sync" \
    --export-root ".../csv6sync"

# or a single pfd/telemetry pair
python3 code/pfd_sync/pfd_compare_csv6.py --pfd <pfd_clean.csv> --data <Elements_0.csv>
```

### 4. OCR alone (what step 1 calls internally)

```bash
python3 code/pfd_sync/pfd2.py <video> --rois <rois.json> --out <pfd_raw.csv>          # default settings
python3 code/pfd_sync/pfd2.py <video> --rois <rois.json> --out <pfd_raw.csv> --preset fast
```

---

## The process

```
PFD video ──► pfd2.py (OCR) ──► pfd_raw.csv
                                     │
                         data_pfd_cleaner.py
                                     │
                               pfd_clean.csv ──┬─► pfd_compare_csv6.py ──► t_scale, t_shift
                                                │        (heading fit)            │
                           telemetry Elements.csv        (this IS the sync)        │
                                                │                                  │
                                                ├─► pfd_verify_airspeed.py ◄───────┤
                                                │        (independent check)      │
                                                └─► pfd_verify_msl_altitude.py ◄───┘
                                                         (independent check)
                                                             │
                                                      summary.json (per cadet)
                                                             │
                                                      pfd_report.py ──► pfd_report.html
```

Step by step, per cadet:

1. **Discover.** `pfd_csv_process.py` finds every `*.c<N>.<name>[_exp_vN]`
   folder under `--video-root`, matches `c<N>` to a sortie number, finds
   the `F2-GARMIN_PFD*` video inside, and looks up that sortie's
   telemetry CSV under `--csv-root`. Anything that can't be matched
   either way (no video for a sortie, or a video session with no
   telemetry) gets recorded, not silently dropped.
2. **OCR.** `pfd2.py` reads 5 fixed screen regions per frame (heading,
   altitude, radar altitude, torque, airspeed) via OCR, skipped entirely
   if `pfd_raw.csv` already exists for that cadet.
3. **Clean.** `data_pfd_cleaner.py` removes physically-implausible
   jumps — the airspeed/altitude readouts are rolling drums that
   occasionally misread mid-roll — using a wide-window outlier filter
   sized from that channel's real telemetry rate of change.
4. **Sync.** `pfd_compare_csv6.py` fits `data_t = t_scale·pfd_t + t_shift`
   on the cleaned heading against telemetry heading, with a wrap-safe
   direct-angle score (not on the derivative — see below for why that
   matters).
5. **Verify.** `pfd_verify_airspeed.py` and `pfd_verify_msl_altitude.py`
   check that same `(t_scale, t_shift)` against two unrelated channels.
   Agreement across all three is what actually earns a `CONFIRMED`.
6. **Export.** the cleaned OCR csv, a copy of telemetry with PFD-clock
   columns inserted, 3 graphs, and one `summary.json` — per cadet.
7. **Report.** `pfd_report.py` turns every cadet's `summary.json` (plus
   the run-wide manifest) into one self-contained HTML file.

---

## Functions, in the order they run

**`pfd2.py`** (OCR)
- `load_rois()` / `px_box()` — merge a `--rois` override onto the
  built-in fractional boxes, resolve to pixel coordinates for this
  video's resolution
- `read_int()` — upscale one crop, OCR it, keep only digits, range-check
- main loop — decode frames from an `ffmpeg` pipe; per field, skip OCR
  if the crop barely changed since last time (`--still`), reuse the
  last value otherwise; `--frame-stride`/`--time-hz` (not used by
  default) additionally skip whole frames

**`data_pfd_cleaner.py`**
- `clean_channel()` — range-clip → Hampel filter (reject vs. a rolling
  median over a window wide enough to outlast the longest observed bad
  OCR run) → frame-to-frame rate gate → interpolate through gaps
- `clean_file()` — runs `clean_channel()` per field, writes the cleaned
  csv, returns a `{field: {kept, raw}}` report

**`pfd_compare_csv6.py`** (the sync)
- `load_pfd()` / `load_data()` — parse the OCR csv / telemetry csv
  (heading columns only)
- `fit()` — coarse-to-fine search over `t_scale`; for each candidate,
  `_FastShiftFinder.best_shift()` finds the best `t_shift` via one FFT
  cross-correlation (the telemetry side's transform is computed once
  and reused across every candidate — the actual speedup over csv5)
- `_score()` — wrap-safe mean-cosine of the angular difference at a
  given `(t_scale, t_shift)` — `1.0` = identical
- `run_batch()` — the `--reuse-root` driver: finds every
  `<label>/pfd_clean.csv` under an existing export, re-fits + verifies
  + graphs + exports each, without touching OCR

**`pfd_verify_airspeed.py`** / **`pfd_verify_msl_altitude.py`**
- `load_pfd_*()` / `load_data_*()` — parse the relevant column (airspeed
  in kt; altitude in ft, telemetry `AltitudeMSL` m→ft)
- `fit_at()` — evaluate correlation at a *given* `(t_scale, t_shift)`
  (used to verify a fit found elsewhere); `fit()` also exists for an
  independent from-scratch search
- `render()` — before/after PNG

**`pfd_csv_process.py`** (the orchestrator)
- `discover_cadet_sessions()` — find cadet folders, extract sortie
  number, locate the PFD video
- `find_telemetry_csv()` — locate `Elements[_0].csv` for a sortie
- `process_cadet()` — runs OCR → clean → sync → verify → export for one
  cadet, wrapped in `try/except` so one failure doesn't kill the batch
- `recommend()` — `CONFIRMED` / `LIKELY OK` / `NEEDS REVIEW` from the
  three scores
- `render_heading()` / `write_synced_telemetry()` — the heading PNG and
  the telemetry-with-PFD-time-columns csv
- `write_summary_csv()` — one row per cadet, this run's record
- `main()` — parses args, cross-checks csv-root sorties vs. video
  sorties (flags anything missing on either side), runs every cadet,
  writes `_summary.csv` + `_run_manifest.json`

**`pfd_report.py`**
- `load_results()` — reads `_run_manifest.json` + every `summary.json`
  under an export dir (falls back to scanning if the manifest is stale)
- `build_errors_section()` — missing video, missing telemetry, skipped
  or errored cadets, up front
- `build_summary_table()` / `build_cadet_section()` — the table and the
  per-cadet detail (graphs embedded as base64 PNG)
- `write_html_report()` — assembles the one-file HTML

---

## File status

| file | role | status |
|---|---|---|
| `pfd2.py` | Same I/O as `pfd.py`; adds `--time-hz`/`--frame-stride` (skip frames — measurably hurts accuracy) and `--scale`/`--still` (cheapen each check — barely moves accuracy). `--preset fast` = `--scale 2 --still 6.0`. | **in use** — default OCR step |
| `pfd.py` | Original OCR tool, full quality. | **available via `--legacy-ocr`** |
| `data_pfd_cleaner.py` | Removes rolling-drum OCR misreads (see above). | **in use** |
| `pfd_compare_csv6.py` | Heading offset+stretch fit, ~3-5x faster than csv5 (FFT reuse), plus `--reuse-root` batch mode. | **in use** |
| `pfd_compare_csv5.py` | Same fit, slower. Superseded by csv6 for anything performance-sensitive; kept as the simpler reference implementation. | not called by the pipeline |
| `pfd_verify_airspeed.py` | PFD airspeed vs. telemetry `TrueAirSpeed`. | **in use** |
| `pfd_verify_msl_altitude.py` | PFD altitude vs. telemetry `AltitudeMSL`. | **in use** |
| `pfd_verify_altitude.py` | PFD RA vs. telemetry `Point_1_HOT` — a semantically worse match (HOT doesn't track RA during a climbing turn away from its reference point), caps real correlations around 0.4-0.8 even when correct. | superseded by `pfd_verify_msl_altitude.py` |
| `pfd_csv_process.py` | Orchestrator: OCR → clean → sync → verify → export, every cadet in one run. | **in use** |
| `pfd_report.py` | `_run_manifest.json` + `summary.json`s → one HTML report. | **in use** |
| `pfd_heading_compare_plot.py` / `_png.py` | Standalone: plot actual heading (not rate) before/after alignment. Imports `pfd_compare_csv3.py`. | ad hoc visualization, not pipeline-wired |
| `pfd_compare_csv.py` / `csv2.py` / `csv3.py` / `csv4.py` | Earlier iterations — per-block local offset, altitude-based fit, heading-*rate*-based fit, heading-direct-but-shift-only fit respectively. csv3's rate-based score reads "weak" even on a visually perfect sync (confirmed: 0.44 on rate vs. 0.99 direct, same data) — the reason csv5/6 exist. | legacy, not used (csv3 is still called by `code/data_flow.py`, a separate older orchestrator not yet updated to this pipeline) |
| `pfd_rois_check1_1024x768.json` | Hand-marked, OCR-verified ROI boxes for this dataset's 1024x768 PFD layout. | **in use** — default `--rois` |

---

## OCR speed: pfd.py vs. pfd2.py

OCR is >99% of the whole pipeline's wall time (~1,400s of ~1,410s for one
sortie) — nothing else comes close, so it's the only stage worth
speeding up. Benchmarked 2026-09-13 on a 5-min high-maneuvering window
(worst case for skip-based savings) from sortie 004, against the
unshortened OCR as ground truth:

| variant | flags | speedup | heading err | altitude err | airspeed err |
|---|---|---|---|---|---|
| pfd2.1 | `--frame-stride 2` | 2.21x | 0.18° | 24.4 ft | 1.0 kt |
| pfd2.2 | `--frame-stride 4` | 2.53x | 0.54° | 55.3 ft | 2.4 kt |
| pfd2.3 | `--time-hz 5` | 3.98x | 2.24° | 133.5 ft | 6.7 kt |
| pfd2.4 | `--time-hz 2` | 7.67x | 3.70° | 195.0 ft | 9.0 kt |
| pfd2.5 | `--scale 3` | 2.17x | 0.06° | 8.8 ft | 0.3 kt |
| pfd2.6 | `--still 4.0` | 2.76x | 0.02° | 4.1 ft | 0.1 kt |
| pfd2.7 | `--time-hz 4 --scale 4` | 4.36x | 2.37° | 156.1 ft | 7.7 kt |
| pfd2.8 | `--scale 3 --still 4.0` | 2.59x | 0.09° | 12.9 ft | 0.3 kt |
| **pfd2.9** | **`--scale 2 --still 6.0`** (**`--preset fast`**) | **3.23x** | **0.14°** | **24.2 ft** | **1.2 kt** |

Two families: **pfd2.1/2.2/2.3/2.4/2.7** decode or OCR *fewer frames* —
skipping a frame outright throws away information a fast-moving
instrument needed, and it shows (0.18°-3.7° heading error). **pfd2.5/
2.6/2.8/2.9** skip nothing — they make each still-check and OCR call
*cheaper* — and accuracy barely moves (0.02°-0.14°). **pfd2.9 won**:
best speedup of the accuracy-preserving family. Re-verified on full
sorties end to end (not just the 5-min clip), against the established
baseline:

| sortie | metric | pfd.py (baseline) | pfd2.py `--preset fast` |
|---|---|---|---|
| 004 | total time | ~1,400s | **816.6s (1.71x)** |
| 004 | heading / airspeed / altitude | 0.9946-0.9951 / 0.930 / 0.987 | 0.9935 / 0.9307 / 0.9869 |
| 005 | total time | 1,116.6s | **737.7s (1.51x)** |
| 005 | heading / airspeed / altitude | 0.9939 / 0.9863 / 0.9719 | 0.9932 / 0.9849 / 0.9729 |

Real-world speedup is lower than the 5-min benchmark's 3.23x because
that window was deliberately picked for heavy maneuvering — worst case
for the still-skip mechanism the baseline already relies on; calmer
cruise segments were already fast at baseline settings, so there's less
left to gain there. `pfd_csv_process.py` uses `pfd2.py --preset fast` by
default; pass `--legacy-ocr` for the old behavior.

---

## A residual worth knowing about: heading has a consistent ~+4.8° offset

Every sortie's best fit (measured 2026-09-13) shows telemetry heading
running **+4.7° to +4.9° above PFD heading** at the same aligned
timestamp — not scattered noise, a near-constant across every cadet:

| sortie | median signed offset | spread (std) |
|---|---|---|
| 002 | +4.69° | 3.78° |
| 003 | +4.92° | 2.69° |
| 004 | +4.72° | 4.32° |
| 005 | +4.78° | 5.23° |
| 006 | +4.90° | 2.63° |
| 008 | +4.75° | 8.41° |

That's the signature of a **magnetic-vs-true heading difference** (or a
fixed instrument calibration offset) — one source is likely magnetic,
the other true, and the delta is this location's magnetic variation.
The current `fit()` only searches `(t_scale, t_shift)`, so it has no way
to correct for this; it just absorbs the ~4.8° into `rms_deg` and
reports it as error. The *spread* column (2.6°-8.4°, varies per sortie)
is the actual OCR-noise component — the sync has been quietly better
than `rms_deg` suggested all along.

**Not yet implemented**: a third fitted parameter (constant angular
bias `b_deg`) would remove this from the error term and report the true
residual noise. Worth doing if `rms_deg` needs to be a meaningful
absolute number rather than just a relative one for comparing sorties.

---

## Output layout (per export root)

```
<export-root>/
  <cadet label>/                  # e.g. 26.1.s1.c4.ilan_x  (library name, no _exp_v2 suffix)
    pfd_raw.csv                    # OCR output, unmodified
    pfd_clean.csv                  # after data_pfd_cleaner.py
    telemetry_synced.csv           # telemetry + "PFD Time (s)"/"PFD Time (ms)" columns
    heading.png / airspeed.png / altitude.png
    summary.json                  # every number above, one place
  _summary.csv                    # one row per cadet, this run's record
  _run_manifest.json              # what was found/skipped/errored + why
  pfd_report.html                 # written by pfd_report.py, not pfd_csv_process.py
```
