"""Shared plumbing for the interchangeable diarization backends
(diarize_pyannote.py, diarize_nemo.py, diarize_speechbrain.py,
diarize_3dspeaker.py, diarize_whisperx.py, diarize_wespeaker.py,
diarize_kaldi.py -- and the original diarize.py).

Every backend does the same two things:
  1. produce speaker-labelled time segments  [(start_s, end_s, raw_id), ...]
  2. hand them to assign_speakers(transcript, segments) here, which maps
     raw ids -> SPEAKER_00.. (busiest speaker first, then first-appearance),
     labels each word by the segment it overlaps most, majority-votes the
     word labels up to segment level, and marks anything uncovered
     SPEAKER_UNK.

So each backend file is just "load model -> get segments -> assign_speakers".
Signature is identical to diarize.py's:  diarize(transcript, wav_path) -> Transcript
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"


def ensure_wav16k(src: str | Path, out: str | Path | None = None) -> Path:
    """Most backends want 16 kHz mono PCM. Re-encode if needed."""
    src = Path(src)
    if out is None:
        out = src.with_suffix(".16k.wav")
    out = Path(out)
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out)],
        check=True,
    )
    return out


def _relabel(segments):
    """raw speaker id -> SPEAKER_00.. ; most talk-time first, ties by first appearance."""
    dur: dict[str, float] = {}
    first: dict[str, float] = {}
    for a, b, spk in segments:
        spk = str(spk)
        dur[spk] = dur.get(spk, 0.0) + max(0.0, b - a)
        first.setdefault(spk, a)
    order = sorted(dur, key=lambda s: (-round(dur[s], 2), first[s]))
    return {spk: f"SPEAKER_{i:02d}" for i, spk in enumerate(order)}


def assign_speakers(transcript, segments):
    """segments: iterable of (start_s, end_s, raw_speaker_id). Returns the
    same transcript with .speaker set on every word and segment."""
    segments = sorted(((float(a), float(b), str(s)) for a, b, s in segments),
                      key=lambda x: x[0])
    if not segments:
        for seg in transcript.segments:
            seg.speaker = "SPEAKER_UNK"
            for w in seg.words:
                w.speaker = "SPEAKER_UNK"
        return transcript

    label = _relabel(segments)

    def at(w0: float, w1: float) -> str:
        best, best_ov = None, 0.0
        for a, b, spk in segments:
            ov = min(w1, b) - max(w0, a)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None:                       # no overlap -> nearest by midpoint
            mid = (w0 + w1) / 2
            best = min(segments, key=lambda s: abs((s[0] + s[1]) / 2 - mid))[2]
            if not (segments[0][0] - 1 <= mid <= segments[-1][1] + 1):
                return "SPEAKER_UNK"
        return label.get(best, "SPEAKER_UNK")

    for seg in transcript.segments:
        for w in seg.words:
            w.speaker = at(w.start, w.end)
        if seg.words:
            vals = [w.speaker for w in seg.words]
            seg.speaker = max(set(vals), key=vals.count)
        else:
            seg.speaker = at(seg.start, seg.end)
    return transcript


def print_summary(transcript, dest):
    counts: dict[str, int] = {}
    for w in transcript.words():
        counts[w.speaker] = counts.get(w.speaker, 0) + 1
    print(f"speakers: {counts} -> {dest}")


def run_cli(backend_name: str, diarize_fn):
    """Standard CLI: `python3 code/speech/diarize_<x>.py transcript.json [--wav ...]`."""
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import Transcript  # noqa: E402

    ap = argparse.ArgumentParser(description=f"Diarization backend: {backend_name}")
    ap.add_argument("transcript_json")
    ap.add_argument("--wav", help="wav path (default exports/audio/<sortie>.wav)")
    args = ap.parse_args()

    t = Transcript.load(args.transcript_json)
    wav = Path(args.wav) if args.wav else AUDIO_DIR / f"{t.sortie}.wav"
    t = diarize_fn(t, wav)
    t.save(args.transcript_json)
    print_summary(t, args.transcript_json)
