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


# default role names by talk-time rank, plus the SIM_* language buckets
_RANK_ROLES = ["trainee / cadet", "instructor", "sim instructor",
               "sim callout", "other speaker", "other speaker"]


def role_breakdown(transcript, roles: dict[str, str] | None = None) -> list[dict]:
    """Per speaker label: talk time (s), % of total talk time, word count,
    and a role name. Roles default to a talk-time-rank heuristic
    (rank 0 -> trainee, 1 -> instructor, 2 -> sim instructor, rest -> sim
    callout) plus SIM_EN/SIM_HE from diarize_langid; pass `roles`
    ({'SPEAKER_00': 'instructor', ...}) to override."""
    secs: dict[str, float] = {}
    words: dict[str, int] = {}
    for seg in transcript.segments:
        spk = seg.speaker or "SPEAKER_UNK"
        for w in (seg.words or [seg]):
            secs[spk] = secs.get(spk, 0.0) + max(0.0, w.end - w.start)
        words[spk] = words.get(spk, 0) + len(seg.words)
    total = sum(secs.values()) or 1e-9

    ranked = [s for s in sorted(secs, key=lambda s: -secs[s])
              if not s.startswith("SIM_") and s != "SPEAKER_UNK"]
    auto = {s: _RANK_ROLES[min(i, len(_RANK_ROLES) - 1)] for i, s in enumerate(ranked)}

    out = []
    for spk in sorted(secs, key=lambda s: -secs[s]):
        if roles and spk in roles:
            role = roles[spk]
        elif spk == "SIM_EN":
            role = "sim callout (EN)"
        elif spk.startswith("SIM_"):
            role = f"sim callout ({spk[4:]})"
        elif spk == "SPEAKER_UNK":
            role = "unassigned"
        else:
            role = auto.get(spk, "other speaker")
        out.append({"speaker": spk, "role": role,
                    "seconds": round(secs[spk], 1),
                    "pct": round(100 * secs[spk] / total, 1),
                    "words": words.get(spk, 0)})
    return out


def format_roles(transcript, roles=None) -> str:
    rows = role_breakdown(transcript, roles)
    w = max((len(r["role"]) for r in rows), default=4)
    lines = [f"  {'role'.ljust(w)}  {'speaker':<12} {'time':>8} {'pct':>7} {'words':>7}"]
    for r in rows:
        lines.append(f"  {r['role'].ljust(w)}  {r['speaker']:<12} "
                     f"{r['seconds']:>7.1f}s {r['pct']:>6.1f}% {r['words']:>7}")
    return "\n".join(lines)


def print_summary(transcript, dest):
    print(format_roles(transcript))
    print(f"  -> {dest}")


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
