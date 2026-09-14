# agents_claude.md — orientation for a future Claude (or human) who forgot everything

Written: 2026-09-14. If you're reading this in 6+ months with no memory of this
repo, start here.

## What this repo is

`drill_compare` analyzes flight-school simulator sorties: multi-camera cockpit
video + audio recordings of cadet/instructor crews, run through pipelines that
extract PFD (instrument) data, eye-gaze/attention, and — the part this doc
mostly covers — speech transcription, speaker diarization, an aviation
glossary tagger, and 15+ CRM/communication "biomarkers" (observation
statements, action statements, OIAV loop completion, hesitations, etc.).

Source video data lives OUTSIDE this git repo, under:
```
/home/ilan/Documents/flight school/ver_data/<course>/<course>.<sortie>/<course>.<sortie>.extrct_v2/
    <course>.<sortie>.c<N>.<cadet_names>_exp_v2/
        00_F2-AUDIO(...).m4a                       -- dedicated mic feed
        01_F2-GARMIN_PFD COCKPIT.mp4
        02_F2-PILOT CAM COCKPIT-(...).mp4
        03_F2-IG12(...).mp4                         -- outside-world/IG view
        04_F2-NADIV_COCKPIT(...).mp4
        05_F2-GARMIN_MFD COCKPIT.mp4
        06_F2-COPILOT CAM COCKPIT-(...).mp4         -- THE speech-pipeline default camera
```
This git repo (`code/`) is just the processing code; `ver_data/` and the
`26.1.s1.audio/` output library below are NOT tracked in git (too large,
mostly media).

## Git state as of this commit

- Repo: `github.com/ilan1sc/drill_compare` (origin), branch `main`.
- This doc was added alongside `code/speech/process_audio_v1.py` (see git log
  for exact commit). Check `git log --oneline -20` for what's landed since.
- **Watch out**: `gazelab2/venv/` (a full python virtualenv, ~500MB / 3100+
  files) got accidentally `git add`-ed at some point before this session.
  It was unstaged and added to `.gitignore` (`venv/`, `.venv/`) as part of
  this sync — if you see it staged again, someone ran a bare `git add -A`
  inside `gazelab2/`. Don't commit it.
- Other in-flight work visible in this repo but NOT explained in detail
  here (belongs to earlier/other sessions): `code/cockpit_gaze/`,
  `code/pfd_sync/`, `gazelab2/` (eye-gaze-on-PFD experiments, see
  `gazelab2/README.md` and the memory file `MEMORY.md` in repo root for a
  pointer to an ATM/face-cam sync memory note).

## The speech/audio pipeline — how it evolved

Full history and the engine bake-off results: **`code/speech/audio_res.md`**
(written 2026-09-06) — read that first if you want the "why", this doc is
just the "where things are now".

Timeline, roughly:
1. `code/speech_v2/pipeline.py` (2026-09-04) — first attempt, ivrit-ai ASR +
   old MFCC diarizer (`code/speech/diarize.py`). Superseded.
2. 2026-09-05 — full bake-off of ASR backends (`transcribe_ivrit_v3/yi/
   turbo/whisper_openai/mms/seamless.py`) and diarization backends
   (`diarize_nemo/whisperx/kaldi/wespeaker/pyannote/3dspeaker/speechbrain/
   langid.py`), scored by `transcribe_compare.py` / `diarize_compare.py`.
3. `code/speech/pipeline_best.py` (2026-09-06) — assembled from the winners.
   **Winners, still current as of this doc:**
   - ASR: **`ivrit-ai/whisper-large-v3`** (Hebrew-finetuned Whisper, via
     `transcribe_ivrit_v3.py`) — best coverage, least hallucination/looping.
   - Diarization: **3D-Speaker / CAM++** (Alibaba, via ModelScope, in
     `diarize_3dspeaker.py`) — cleanest 2-speaker split, most consistent
     across sorties; can self-pick speaker count (`--speakers auto`).
   - Glossary: `code/speech/glossary.py` (aviation term tagging, v2).
   - Biomarkers: `code/biomarkers/b01..b15_*.py`, run via
     `code/speech/run_biomarkers.py`.
4. `code/speech/audio_vis_biomarkers.py` + `av_sync.py` (2026-09-07) — the
   "augmented movie" renderer: burns in a cadet/instructor speaker-color
   banner, RTL Hebrew subtitles, and a live right-edge panel that lights up
   each biomarker (b01..b15) as it fires on the current utterance, with
   running counts. Includes automatic A/V sync verification+correction
   (random-sampled offset check, reported in a `.sync.json` next to the mp4).
5. **`code/speech/process_audio_v1.py`** (2026-09-13/14, this session) — the
   thing you should actually run. Packages steps 3+4 into one CLI command:
   trim a window (or the whole file) → transcribe → diarize → glossary-tag →
   biomarkers → (optionally) render the augmented movie. Superset of
   `pipeline_best.py` with a nicer CLI; `pipeline_best.py` and
   `speech_v2/pipeline.py` are effectively legacy at this point but still
   present/runnable for reference.

### How to run it

```bash
cd "/home/ilan/Documents/flight school/drill_compare"

# whole sortie (no --duration => probes file length, runs full length)
python3 code/speech/process_audio_v1.py \
    "/home/ilan/Documents/flight school/ver_data/26.1/26.1.s1/26.1.s1.extrct_v2/26.1.s1.c6.sorek_bigeyes_exp_v2" \
    --out-dir "/home/ilan/Documents/flight school/ver_data/26.1/26.1.s1/26.1.s1.audio/26.1.s1.c6.sorek_bigeyes"

# short sample (fast iteration / testing)
python3 code/speech/process_audio_v1.py "<cadet source dir>" \
    --out-dir "<cadet library dir>" --start 600 --duration 60

# skip the movie render (much faster — just transcript+biomarkers)
python3 code/speech/process_audio_v1.py "<cadet source dir>" \
    --out-dir "<cadet library dir>" --skip-movie
```

Key facts about this script:
- First positional arg can be an exact video file, OR a cadet's source dir
  (auto-picks `06_F2-COPILOT CAM COCKPIT*` — override with `--cam-glob`).
- `--audio-source` lets you use the dedicated mic feed
  (`00_F2-AUDIO(...).m4a`) instead of the video's own audio track.
- The trimmed/full video used to render the movie is a TEMP file
  (`/tmp/process_audio_v1_*.mp4`), deleted after rendering — the library
  never keeps a copy of source video. This was an explicit fix requested
  mid-session (previously it left `sample_video.mp4` copies behind).
- Output layout per run, under `<out-dir>/<sample-name>/` (sample-name
  defaults to `sample_<duration>s` or `sample_full`):
  ```
  transcript.json                  -- schema.py Transcript: segments/words/speakers
  glossary_review_queue.csv        -- low-confidence glossary matches to review
  biomarkers/b01_*.json ... b15_*.json, _summary.json
  <name>.wav, <name>.16k.wav        -- kept (NOT auto-deleted)
  biomarker_vis_<start>_<dur>.mp4  -- only if movie wasn't skipped
  biomarker_vis_<start>_<dur>.sync.json
  ```
- Runs cleanly in parallel across cadets (3-way tested) — GPU here is an
  NVIDIA GB10 with 124GB unified memory, plenty of headroom; the ASR+diarize
  models each load fine per-process. CPU-bound ffmpeg trim/encode and the
  frame-by-frame movie render are the actual bottleneck, not GPU memory.
- Full-length movie rendering is SLOW: roughly the same order of magnitude
  as the video's own length (a ~45-50min sortie took on the order of an
  hour to render full-length, running 3 in parallel). Budget for it, or use
  `--skip-movie` / a short `--start/--duration` window for quick looks.

## The output library: `26.1.s1.audio/`

```
/home/ilan/Documents/flight school/ver_data/26.1/26.1.s1/26.1.s1.audio/
    26.1.s1.c1.rothman_yesay/sample_2982s/...
    26.1.s1.c2.ilan_oria/sample_2720s/...
    26.1.s1.c3.rothman_rotkof/sample_2720s/...
    26.1.s1.c4.ilan_x/sample_2892s/...
    26.1.s1.c5.oren_efrati/sample_2334s/...
    26.1.s1.c6.sorek_bigeyes/sample_1min/, sample_60s/   -- SHORT SAMPLES ONLY,
                                                             c6 was explicitly
                                                             skipped in the
                                                             full-sortie batch
    26.1.s1.c7.oren_noam/sample_3232s/...
    26.1.s1.c8.sorek_lifman/sample_2878s/...
    _run_logs/                        -- stdout/stderr logs from the batch runs
```

This mirrors the reference layout that was already established in
`exports/audio_best/<sortie>/` (the earlier single-sortie naming scheme, git-
ignored, still present from the 2026-09-06/07 work) — same file set, just
reorganized per-cadet under `ver_data/` instead of per-sortie under
`exports/`.

### Full-sortie batch run (2026-09-14)

Ran `process_audio_v1.py --skip-movie` (transcript+diarize+glossary+
biomarkers only) on all 7 cadets except c6 (explicitly asked to skip it),
3-way parallel. All 7 completed successfully (exit=0). Result snapshot —
obs/min, act/min, comm-relevance share (see each cadet's
`biomarkers/_summary.json` for the full 15-biomarker breakdown):

| cadet | length | obs/min | act/min | comm-relevance |
|---|---|---|---|---|
| c1 rothman_yesay | 2982s | 2.17 | 2.11 | 0.53 |
| c2 ilan_oria | 2720s | 0.40 | 0.11 | 0.29 |
| c3 rothman_rotkof | 2720s | 3.35 | 0.91 | 0.43 |
| c4 ilan_x | 2892s | 4.21 | 2.28 | 0.58 |
| c5 oren_efrati | 2334s | 2.24 | 1.29 | 0.46 |
| c7 oren_noam | 3232s | 3.09 | 1.49 | 0.47 |
| c8 sorek_lifman | 2878s | 0.54 | 0.10 | 0.27 |

**c2 and c8 are notable low outliers** (obs/act rates ~5-10x lower than the
rest, comm-relevance ~0.27-0.29 vs ~0.43-0.58 elsewhere) — worth checking
whether that's a real behavioral signal or a transcription/diarization
quality issue before drawing conclusions. `audio_res.md` flagged a similar
low-outlier pattern for c2 back in the original single-sortie bake-off, so
this may be a real, reproducible property of that cadet's recording (mic
placement? a genuinely quiet crew?) rather than a pipeline bug — but it
hasn't been root-caused.

At the time this doc was written, a follow-up batch to render the FULL-
LENGTH augmented movies (`biomarker_vis_0_<duration>.mp4`) for all 7 was
**still running in the background** (started ~23:11, 3-way parallel,
c1/c2/c3 first, then c4/c5/c7/c8 queued). Check `26.1.s1.audio/_run_logs/
movie_<cadet>.log` and look for `biomarker_vis_0_*.mp4` files in each
cadet's sample dir to see whether it finished; if a cadet is missing one,
it either didn't finish or wasn't started yet — the command to (re)run it
for one cadet is in "How to run it" above (matching `--start 0 --duration
<full length>`, full lengths are in the table above / each `transcript.json`
`"duration"` field).

## If you want to extend this

- **c6** only has short samples (`sample_1min`, `sample_60s`) — never got a
  full-sortie run. Source dir:
  `26.1.s1.extrct_v2/26.1.s1.c6.sorek_bigeyes_exp_v2/`.
- To re-run any cadet from scratch, just re-invoke `process_audio_v1.py` —
  it's idempotent (overwrites the sample dir's contents), nothing to clean
  up first except old `.wav`s if you're low on disk.
- The `.wav` / `.16k.wav` copies in each sample dir are NOT auto-deleted
  (unlike the video, which is). If disk gets tight, they're safe to delete
  after `transcript.json` exists — nothing downstream re-reads the audio
  except biomarker b15 (acoustic candidates) and the movie renderer, both
  of which only run once per invocation.
- `--speakers auto` (instead of the default `2`) lets 3D-Speaker pick the
  speaker count itself — worth trying if a cadet's diarization looks
  collapsed/wrong (this happened to `s1 c5` in the original 2026-09-06 run,
  per `audio_res.md`'s "Known issues" section — not the same c5 as
  `26.1.s1.c5.oren_efrati` above, different course/sortie numbering, but
  the same failure mode is possible here too).
- If you're not in this git repo but reading this file, the actual code
  paths above are relative to `<repo_root>/code/speech/`.
