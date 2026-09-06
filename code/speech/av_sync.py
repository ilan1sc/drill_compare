"""Deterministic A/V muxing + an automatic sync verifier/corrector for the
audio_vis / biomarker movies.

Why: the old path (write a silent .mp4 of CAP_PROP_FRAME_COUNT frames, then
`-shortest` mux against a separately-cut audio clip) drifts -- CAP_PROP_FRAME_COUNT
is unreliable and the render loop can stop early, so the video plays for
less wall-time than the audio and they walk apart.

Fixes here:
  accurate_cfr()   -- constant-fps VIDEO-ONLY clip cut with an ACCURATE seek
                      (`-ss` AFTER `-i`): frame i is exactly at start + i/fps.
  render_silent()  -- pipe rendered BGR frames straight into ffmpeg at a fixed
                      `-r fps`; you pass exactly round(duration*fps) frames
                      (pad with the last frame), so length is exact.
  add_audio()      -- attach the SOURCE audio from the same accurate `-ss`
                      (optionally shifted by `audio_offset`); `-c:v copy`, fast,
                      repeatable.
  verify()         -- sample random points along the finished movie and at each
                      measure the VIDEO offset (match the movie frame against
                      source frames in a +/-1s window by normalised cross-
                      correlation, parabolic sub-frame refine) and the AUDIO
                      offset (GCC-PHAT between movie and source audio windows).
  render_and_sync()-- render_silent -> add_audio -> verify; if |median audio
                      offset| exceeds a threshold, re-run add_audio once with a
                      compensating shift and re-verify. Fully automatic.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def accurate_cfr(video, start: float, duration: float, fps: float) -> Path:
    out = Path(tempfile.gettempdir()) / f"cfr_{abs(hash((str(video), start, duration, fps)))}.mp4"
    _run(["ffmpeg", "-y", "-v", "error",
          "-ss", f"{start}", "-i", str(video), "-t", f"{duration}",
          "-an", "-r", f"{fps}", "-vsync", "cfr",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
          "-pix_fmt", "yuv420p", str(out)])
    return out


def render_silent(frames, w: int, h: int, fps: float, out_path) -> tuple[Path, int]:
    """frames: iterator of HxWx3 uint8 BGR arrays, one per 1/fps, in order."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
         str(out_path)],
        stdin=subprocess.PIPE)
    n = 0
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        n += 1
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg render_silent failed")
    return out_path, n


def add_audio(silent, source, start: float, duration: float, out_path,
              audio_offset: float = 0.0) -> Path:
    """audio_offset > 0 delays the audio (use when audio currently LEADS);
    < 0 advances it (audio currently LAGS)."""
    out_path = Path(out_path)
    _run(["ffmpeg", "-y", "-v", "error",
          "-i", str(silent),
          "-ss", f"{max(0.0, start - audio_offset)}", "-i", str(source),
          "-t", f"{duration}",
          "-map", "0:v:0", "-map", "1:a:0?",
          "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)])
    return out_path


# ---------------------------------------------------------------- verify ----
def _frame(video, t: float):
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        png = f.name
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t)}", "-i", str(video),
          "-frames:v", "1", png])
    im = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    Path(png).unlink(missing_ok=True)
    return im


def _clip_gray(video, t0: float, dur: float, fps: float = 25.0):
    """One ffmpeg call -> list of (rel_time, grayscale frame) for a short span."""
    import cv2
    tmp = Path(tempfile.gettempdir()) / f"vseg_{abs(hash((str(video), t0, dur)))}.mp4"
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t0)}", "-i", str(video),
          "-t", f"{dur}", "-an", "-r", f"{fps}", "-vsync", "cfr",
          "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", str(tmp)])
    cap = cv2.VideoCapture(str(tmp))
    out, k = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append((k / fps, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
        k += 1
    cap.release()
    tmp.unlink(missing_ok=True)
    return out


def _wav(video, t: float, dur: float, sr: int = 16000):
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t)}", "-i", str(video),
          "-t", f"{dur}", "-ac", "1", "-ar", str(sr), wav])
    y, _ = sf.read(wav)
    Path(wav).unlink(missing_ok=True)
    return np.asarray(y, np.float32)


def _ncc(a, b):
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float((a * b).sum() / d)


def _gcc_phat(sig, ref, sr, max_lag_s=0.5):
    n = 1 << int(np.ceil(np.log2(len(sig) + len(ref))))
    S = np.fft.rfft(sig, n)
    R = np.fft.rfft(ref, n)
    G = S * np.conj(R)
    G /= np.abs(G) + 1e-9
    cc = np.fft.irfft(G, n)
    m = int(max_lag_s * sr)
    cc = np.concatenate([cc[-m:], cc[:m + 1]])
    k = int(np.argmax(np.abs(cc)))
    return (k - m) / sr            # >0: sig lags ref


def verify(movie, source, start: float, duration: float, n: int = 12,
           seed: int = 0, fps: float = 25.0) -> dict:
    """video_offset_s > 0 => movie video is BEHIND the source (earlier moment).
    audio_offset_s  > 0 => movie audio LAGS the source."""
    import cv2
    rng = np.random.default_rng(seed)
    ts = sorted(float(x) for x in rng.uniform(2, max(3, duration - 2), size=n))
    search, sfps = 0.8, min(fps, 30.0)          # +/-0.8s window, capped search resolution
    v_errs, a_errs, rows = [], [], []
    for t in ts:
        mv = _frame(movie, t)
        if mv is None:
            continue
        H, W = mv.shape
        clip = _clip_gray(source, start + t - search, 2 * search, sfps)   # 1 ffmpeg call
        scores = {}
        for rel, sfr in clip:
            dt = (rel - search)                 # source time = (start+t) + dt
            sfr = cv2.resize(sfr, (min(W, sfr.shape[1]), H))
            wv = sfr.shape[1]
            cw, ch = int(wv * 0.5), int(H * 0.5)
            x0, y0 = (wv - cw) // 2, (H - ch) // 2
            scores[round(dt, 4)] = _ncc(mv[y0:y0 + ch, x0:x0 + cw],
                                        sfr[y0:y0 + ch, x0:x0 + cw])
        if not scores:
            continue
        keys = sorted(scores)
        best_dt = max(scores, key=scores.get)
        if best_dt not in (keys[0], keys[-1]):
            i = keys.index(best_dt)
            y1, y2, y3 = scores[keys[i - 1]], scores[keys[i]], scores[keys[i + 1]]
            den = y1 - 2 * y2 + y3
            if abs(den) > 1e-9:
                best_dt += 0.5 * (y1 - y3) / den * (1.0 / sfps)
        v_off = -best_dt

        ma = _wav(movie, t, 2.0)
        sa = _wav(source, start + t, 2.0)
        m = min(len(ma), len(sa))
        a_off = _gcc_phat(ma[:m], sa[:m], 16000) if m > 1600 else float("nan")

        v_errs.append(v_off)
        if np.isfinite(a_off):
            a_errs.append(a_off)
        rows.append({"t": round(t, 2), "video_offset_s": round(v_off, 3),
                     "audio_offset_s": round(float(a_off), 3) if np.isfinite(a_off) else None,
                     "video_match": round(scores[max(scores, key=scores.get)], 3)})

    def stat(x):
        x = np.asarray(x, float)
        if not len(x):
            return {"median": None, "mad": None, "n": 0}
        med = float(np.median(x))
        return {"median": round(med, 3),
                "mad": round(float(np.median(np.abs(x - med))), 3), "n": int(len(x))}

    return {"movie": str(movie), "points": rows,
            "video_offset": stat(v_errs), "audio_offset": stat(a_errs)}


def render_and_sync(frames, w, h, fps, source, start, duration, out_path,
                    auto=True, tol_s=0.04, verify_n=12, seed=0, log=print):
    """render_silent -> add_audio -> verify; if auto and audio is out by
    more than tol_s, re-mux once with a compensating shift and re-verify."""
    out_path = Path(out_path)
    silent = out_path.with_name(out_path.stem + ".silent.mp4")
    render_silent(frames, w, h, fps, silent)
    add_audio(silent, source, start, duration, out_path, audio_offset=0.0)

    rep = verify(out_path, source, start, duration, n=verify_n, seed=seed, fps=fps)
    log(f"[sync] video offset {rep['video_offset']['median']}s "
        f"(MAD {rep['video_offset']['mad']}, n={rep['video_offset']['n']})  |  "
        f"audio offset {rep['audio_offset']['median']}s "
        f"(MAD {rep['audio_offset']['mad']}, n={rep['audio_offset']['n']})")
    a_med = rep["audio_offset"]["median"]
    if auto and a_med is not None and abs(a_med) > tol_s:
        log(f"[sync] correcting audio by {a_med:+.3f}s and re-checking ...")
        add_audio(silent, source, start, duration, out_path, audio_offset=a_med)
        rep2 = verify(out_path, source, start, duration, n=verify_n, seed=seed + 1, fps=fps)
        log(f"[sync] after: video {rep2['video_offset']['median']}s  |  "
            f"audio {rep2['audio_offset']['median']}s")
        rep = {"before": rep, "after": rep2, "applied_audio_offset_s": a_med}
    silent.unlink(missing_ok=True)
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Verify A/V sync of a rendered movie vs its source")
    ap.add_argument("movie")
    ap.add_argument("source")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args()
    r = verify(a.movie, a.source, a.start, a.duration, a.n, a.seed, a.fps)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\nvideo offset  median {r['video_offset']['median']}s  (MAD {r['video_offset']['mad']}s)")
    print(f"audio offset  median {r['audio_offset']['median']}s  (MAD {r['audio_offset']['mad']}s)")
