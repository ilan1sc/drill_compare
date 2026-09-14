# cockpit_gaze

Projects the cadet's gaze (extracted from the copilot-cam face video,
06) onto the pilot-cam cockpit video (02), so you can see where on the
panel they were actually looking, and debug the gaze algorithm itself.

## Files

- `pfd_finder.py` — locates the PFD screen's pixel box in a 02 video
  (leftmost of the two bright Garmin displays in the centre console;
  the rig moves between sessions so this is detected per-video, not
  hardcoded). `--preview` dumps a frame with the box drawn in red.
- `cockpit_gaze_pipeline.py` — the full pipeline: extract gaze (06, left
  seat, crop+no-upscale+no-enhance per `gazelab2`'s findings) → cluster
  into on_pfd/other_inside/outside (reuses `code/atm/atm_gaze_cluster.py`)
  → locate PFD (`pfd_finder`) → project → render both augmented movies +
  the heatmap + the per-frame CSV.

## Sync

Both cameras come from the same recording session, and
`manifest.json` carries each stream's wall-clock `start_unix` — so the
06↔02 time offset is exact, no OCR needed: `t02 = t06 + (start06 - start02)`
(+0.628s for `26.1.s1.c6`).

## The "homography"

Not a true 4-point planar homography — there's no shared plane between
a face-on camera and a from-behind cockpit camera. Instead (same model
as the existing ATM-camera gaze work): the auto-detected on-PFD gaze
cluster's centroid (yaw₀, pitch₀) is pinned to the PFD box's pixel
centre, and degrees convert to pixels via an assumed camera FOV
(`--hfov`/`--vfov`, default 90°/70°, uncalibrated — tune if the dot
doesn't track a known deliberate glance correctly):

```
x = pfd_cx - flip_x · sx · (yaw - yaw0)      sx = frame_w / hfov
y = pfd_cy - sy · (pitch - pitch0)           sy = frame_h / vfov
```

`flip_x` defaults **on**: the copilot cam (06, faces the crew) and the
pilot cam (02, faces the panel from behind the crew) are mirrored
relative to each other, so a cadet gaze-yaw sign in one is reversed in
the other. `--no-flip-x` to disable if a run looks mirrored.

Saved to `calibration.json` per cadet, alongside the raw gaze track,
the clustered labels, the per-frame projection, both augmented movies,
and the static heatmap.

## Known limitations / not yet done

- FOV (`--hfov`/`--vfov`) is a guess, not measured from the lens. The
  heatmap on `26.1.s1.c6` (0-120s) looks qualitatively right (hot spot
  on the PFD box, plausible spread onto the windscreen for "outside"
  glances) but the absolute pixel accuracy of anything off the PFD
  hasn't been validated against ground truth.
- Only run over short windows so far (a couple of minutes), not full
  sorties — the pipeline takes `--start`/`--duration` and generalizes
  directly, just untested at full (~40 min) length.
- Only tested on `26.1.s1.c6`. `pfd_finder.py`'s brightness heuristic
  should generalize (that's the whole point — no hardcoded PFD
  position) but hasn't been run across the other 7 cadets yet.
- `flip_x=True` is a physical-plausibility assumption (mirrored camera
  mounts), not yet confirmed against a moment with an unambiguous known
  gaze direction (e.g., cadet looking hard left/right at a real target
  visible in both cameras).

## Usage

```bash
python3 code/cockpit_gaze/cockpit_gaze_pipeline.py \
    --cadet-dir ".../26.1.s1.extrct_v2/26.1.s1.c6.sorek_bigeyes_exp_v2" \
    --start 0 --duration 300
```

Needs the `gazelab2/venv` environment (mediapipe + compatible
protobuf) — the repo's global `protobuf` is too new for `mediapipe`
0.10.18's `FaceLandmarker`:

```bash
gazelab2/venv/bin/python code/cockpit_gaze/cockpit_gaze_pipeline.py ...
```
