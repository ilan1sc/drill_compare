"""12.3 Secondary acoustic candidates: pitch variability, energy, rhythm,
jitter and shimmer -- PRD explicitly flags these as requiring validation
before use in assessment; this module extracts them and stops there. Do
not feed these into a score without a validation pass.

Per cadet segment (from --wav, the 16kHz mono audio extract_audio.py made):
  - pitch (F0, Hz)  via librosa.pyin
  - energy (RMS)     via librosa.feature.rms
  - rhythm            words-per-second per segment (speaking-rate proxy)
  - jitter / shimmer  ** crude frame-to-frame proxies **, NOT the standard
                      pitch-period-based jitter/shimmer (that needs
                      praat-parselmouth, not installed here): jitter_proxy
                      = mean relative |deltaF0| between voiced frames;
                      shimmer_proxy = mean relative |deltaRMS|.

Aggregated to sortie-level mean/std plus a per-segment table for anyone
who wants to correlate these against the other biomarkers or ground truth.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult, cadet_segments, run_cli  # noqa: E402

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "exports" / "audio"


def _segment_acoustics(y, sr, t0, t1):
    import librosa
    i0, i1 = int(t0 * sr), max(int(t0 * sr) + 1, int(t1 * sr))
    clip = y[i0:i1]
    if len(clip) < sr * 0.2:
        return None
    f0, voiced, _ = librosa.pyin(clip, fmin=60, fmax=500, sr=sr)
    f0v = f0[voiced] if voiced is not None else f0[~np.isnan(f0)]
    f0v = f0v[~np.isnan(f0v)] if f0v is not None else np.array([])
    rms = librosa.feature.rms(y=clip)[0]

    jitter = shimmer = None
    if len(f0v) > 3:
        d = np.abs(np.diff(f0v)) / f0v[:-1]
        jitter = float(np.mean(d))
    # near-silent frames blow up the relative-difference ratio (dividing by
    # ~0); only use frames with real energy, floored at 1% of this clip's peak
    floor = rms.max() * 0.01
    rms_v = rms[rms > floor]
    if len(rms_v) > 3:
        d = np.abs(np.diff(rms_v)) / rms_v[:-1]
        shimmer = float(np.mean(d))

    return {
        "pitch_mean_hz": float(np.mean(f0v)) if len(f0v) else None,
        "pitch_std_hz": float(np.std(f0v)) if len(f0v) else None,
        "energy_rms_mean": float(np.mean(rms)),
        "jitter_proxy": jitter,
        "shimmer_proxy": shimmer,
    }


def compute(transcript, cadet_speaker=None, wav_path: str | None = None) -> BiomarkerResult:
    import librosa
    wav_path = wav_path or (AUDIO_DIR / f"{transcript.sortie}.wav")
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)

    rows = []
    for seg in cadet_segments(transcript, cadet_speaker):
        a = _segment_acoustics(y, sr, seg.start, seg.end)
        if a is None:
            continue
        dur = max(seg.end - seg.start, 1e-6)
        a.update({"t": round(seg.start, 2), "text": seg.text,
                  "words_per_s": round(len(seg.words) / dur, 3)})
        rows.append(a)

    def _agg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)} \
            if vals else None

    metrics = {
        "segments_analyzed": len(rows),
        "pitch_mean_hz": _agg("pitch_mean_hz"),
        "pitch_variability_hz": _agg("pitch_std_hz"),
        "energy_rms": _agg("energy_rms_mean"),
        "rhythm_words_per_s": _agg("words_per_s"),
        "jitter_proxy": _agg("jitter_proxy"),
        "shimmer_proxy": _agg("shimmer_proxy"),
        "validated": False,
    }
    return BiomarkerResult("b15_acoustic_candidates", metrics, rows,
                           limitations="UNVALIDATED exploratory signal per PRD 12.3; "
                                       "jitter/shimmer are crude frame-to-frame proxies, "
                                       "not standard pitch-period jitter/shimmer -- see docstring")


if __name__ == "__main__":
    import argparse
    from base import Transcript, guess_cadet_speaker

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript_json")
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--wav", default=None)
    args = ap.parse_args()

    t = Transcript.load(args.transcript_json)
    cadet = args.cadet_speaker or guess_cadet_speaker(t)
    result = compute(t, cadet_speaker=cadet, wav_path=args.wav)
    out = result.save(t.sortie)
    print(f"b15_acoustic_candidates: {result.metrics}")
    print(f"Wrote {out}")
