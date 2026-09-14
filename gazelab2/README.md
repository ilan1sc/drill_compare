# gazelab2 — gaze-detection-accuracy lab

Restricted to the **cadet's seat (left)** — the copilot-cam footage only
shows a usable face on that side; the goal here was to find why
detection quality was so poor and fix it.

## The problem

`exports/gaze/s1 c6/left.json` (from `code/gaze/extract_gaze.py`, current
defaults) has a face detected in only **~9%** of sampled frames over the
whole 45-minute sortie. A gaze "track" that's 91% holes isn't usable.

## Method

`lab.py` reimplements the extractor with every knob exposed, and runs a
battery of configs over the identical 5-minute window (`t=0-300s`) of
`s1 c6.mp4` (1920x1080, 60fps — left seat = left half of frame), so
detection coverage is directly comparable. Metrics:

- **coverage** — detections / frames sampled (the main thing that was broken)
- **jitter_deg_median** — median |Δyaw|+|Δpitch| between consecutive kept
  samples (precision proxy, secondary)
- **wall_s** — cost, since some fixes are much more expensive than others

## Results (5-min window, 1501 sampled frames @5Hz)

| config | coverage | jitter (deg) | wall (s) | note |
|---|---|---|---|---|
| A_baseline | **24.7%** | 7.84 | 61.8 | current `extract_gaze.py` defaults (full frame, both seats) |
| B_left_crop | 97.6% | 3.73 | 41.7 | crop to left half, nothing else changed |
| C_crop_upscale2x | 97.6% | 3.68 | 113.1 | + 2x upscale |
| D_crop_upscale2x_noenhance | 97.9% | 3.33 | 35.4 | + 2x upscale, drop gray-world/CLAHE |
| E_crop_upscale2x_clahe_only | 98.2% | 3.54 | 53.1 | + 2x upscale, CLAHE without gray-world |
| F_crop_upscale2x_lowdet | 97.7% | 3.68 | 116.9 | + 2x upscale, looser mediapipe thresholds |
| G_crop_upscale3x | 97.5% | 3.78 | 204.5 | 3x upscale |
| H_crop_upscale2x_strongclahe | 97.7% | 3.69 | 111.9 | 2x upscale, stronger CLAHE clip |
| **I_crop_noupscale_noenhance** | **98.0%** | **3.39** | **29.8** | crop only, no upscale, no enhance |
| J_crop_noupscale_clahe_only | 98.1% | 3.63 | 30.2 | crop only, CLAHE without gray-world |

## Conclusion

**Cropping to the left seat is the entire fix.** 24.7% → 97.6% coverage
from that alone (config B); every other knob tested (upscale 2x/3x,
enhancement on/off/variant, looser detection thresholds) moves coverage
by at most ±0.6 points — noise, not signal — while upscaling costs
2-5x more wall time for it. The remaining 2% miss rate is presumably
genuine occlusion/motion-blur frames, not a fixable preprocessing gap.

Bonus finding: the gray-world white-balance step in the current
`_enhance()` doesn't help on this footage — **I** (no enhancement at
all) has both better coverage and lower jitter than **B** (which still
ran the full gray-world+CLAHE enhance), and is the fastest config
tested. The green-cast/dim-cabin assumption that motivated that step
doesn't seem to hold for this clip.

**Recommendation**: in `extract_gaze.py`, when a cadet video is known
single-seat, run with `crop=(0, 0, w // 2, h)` (or the actual seat side)
and `seat_override="left"` — skip `num_faces=2` and the x-position seat
heuristic entirely — and drop the gray-world pre-step (keep CLAHE only,
or skip both — I and J are within noise of each other). No upscale
needed. Not yet wired into `extract_gaze.py` itself; this lab only
proves out the fix on `s1 c6`.

## Files

- `lab.py` — the benchmark harness; `--video/--start/--end/--configs`
- `venv/` — isolated venv (repo's global `protobuf` is 7.x, incompatible
  with `mediapipe` 0.10.18's `FaceLandmarker` — this venv pins
  `protobuf<4.24` so mediapipe works; the venv is local to this lab and
  doesn't touch the project's global environment)
- `results/<config>.json` — one GazeTrack per config, same schema as
  `code/gaze/schema.py`
- `results/_comparison.csv` — overwritten per `lab.py` run (only holds
  whichever `--configs` subset was last run — the table above is the
  merged, complete record)

## Reproduce

```bash
cd gazelab2
venv/bin/python lab.py --video "/home/ilan/Documents/flight school/data/s1 c6.mp4" --start 0 --end 300
```
