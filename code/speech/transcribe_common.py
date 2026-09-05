"""Shared plumbing for the interchangeable transcription backends
(transcribe_whisper_openai.py, transcribe_ivrit_v3.py,
transcribe_ivrit_yi.py, transcribe_ivrit_turbo.py, transcribe_mms.py,
transcribe_seamless.py -- and the originals transcribe.py /
speech_v2/transcribe.py).

Each backend is just "load model -> get (word, start, end, prob) list ->
build_transcript()". Signature matches the originals:

    transcribe(video, language="he") -> schema.Transcript

CTC / seq2seq models that don't segment long audio themselves
(MMS, SeamlessM4T) use chunk() here to cut fixed windows first.
metrics() computes the per-model report numbers.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_audio import extract  # noqa: E402
from schema import Segment, Transcript, Word  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"
HEBREW = re.compile(r"[֐-׿יִ-ﭏ]")
LATIN = re.compile(r"[A-Za-z]")
WORDCHAR = re.compile(r"\w", re.UNICODE)


def load_wav(video: str | Path, sr: int = 16000):
    import soundfile as sf
    wav = extract(video, out_dir=AUDIO_DIR) if Path(video).suffix.lower() != ".wav" else Path(video)
    y, in_sr = sf.read(str(wav))
    if y.ndim > 1:
        y = y.mean(1)
    y = y.astype(np.float32)
    if in_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
    return y, sr, Path(wav)


def chunk(y, sr, win_s=25.0, hop_s=24.0):
    """[(t0, samples), ...] fixed windows for models with no long-form mode."""
    out, t = [], 0.0
    n = len(y)
    while t * sr < n:
        i0, i1 = int(t * sr), int(min((t + win_s), n / sr) * sr)
        if i1 - i0 < int(0.2 * sr):
            break
        out.append((t, y[i0:i1]))
        t += hop_s
    return out


def _words_to_segments(words: list[Word], pause_gap: float = 0.6) -> list[Segment]:
    words = [w for w in words if WORDCHAR.search(w.text)]
    segs, cur = [], []
    for w in words:
        if cur and w.start - cur[-1].end > pause_gap:
            segs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    out = []
    for i, ws in enumerate(segs):
        out.append(Segment(id=i, start=ws[0].start, end=ws[-1].end,
                           text=" ".join(w.text for w in ws).strip(), lang="he",
                           speaker=None, words=ws))
    return out


def build_transcript(sortie, video, words: list[Word], segments=None) -> Transcript:
    segs = segments if segments is not None else _words_to_segments(words)
    dur = segs[-1].end if segs else 0.0
    return Transcript(sortie=sortie, source_video=str(video), duration=dur, segments=segs)


_HF_PIPES: dict = {}


def hf_whisper(video, model_id: str, language: str | None = "he") -> Transcript:
    """Shared path for the transformers-hosted Whisper checkpoints
    (openai/whisper-large-v3 and the ivrit-ai fine-tunes). Word timestamps
    via return_timestamps='word'; long-form via chunk_length_s."""
    import torch
    from transformers import pipeline

    y, sr, wav = load_wav(video)
    if model_id not in _HF_PIPES:
        _HF_PIPES[model_id] = pipeline(
            "automatic-speech-recognition", model=model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu")
    gk = {"task": "transcribe"}
    if language:
        gk["language"] = language
    res = _HF_PIPES[model_id]({"array": y, "sampling_rate": sr}, return_timestamps="word",
                              chunk_length_s=28, stride_length_s=4, generate_kwargs=gk)
    words = [Word(text=c["text"].strip(), start=float(c["timestamp"][0]),
                 end=float(c["timestamp"][1] or c["timestamp"][0]), prob=1.0)
            for c in res.get("chunks", []) if c["text"].strip()]
    return build_transcript(Path(video).stem, video, words)


def timed(fn):
    """wrap a backend's transcribe() to also stash .wall_seconds on the result."""
    def wrapper(*a, **k):
        t0 = time.time()
        tr = fn(*a, **k)
        tr._wall_seconds = round(time.time() - t0, 1)
        return tr
    return wrapper


def metrics(transcript: Transcript, audio_seconds: float | None = None) -> dict:
    words = list(transcript.words())
    text_all = " ".join(w.text for w in words)
    n = len(words)
    probs = [w.prob for w in words if w.prob is not None]
    low = [p for p in probs if p < 0.5]

    heb = sum(1 for w in words if HEBREW.search(w.text))
    lat = sum(1 for w in words if LATIN.search(w.text) and not HEBREW.search(w.text))
    junk_segs = sum(1 for s in transcript.segments if not WORDCHAR.search(s.text))

    # hallucination proxy: how repetitive is the output (looped n-grams)
    toks = re.findall(r"\w+", text_all, re.UNICODE)
    tri = list(zip(toks, toks[1:], toks[2:]))
    rep = round(1 - len(set(tri)) / len(tri), 3) if tri else 0.0

    speech_s = sum(s.end - s.start for s in transcript.segments)
    return {
        "words": n,
        "segments": len(transcript.segments),
        "speech_seconds": round(speech_s, 1),
        "coverage": round(speech_s / audio_seconds, 3) if audio_seconds else None,
        "words_per_min": round(n / (speech_s / 60), 1) if speech_s else 0.0,
        "mean_word_prob": round(float(np.mean(probs)), 3) if probs else None,
        "low_conf_word_pct": round(100 * len(low) / len(probs), 1) if probs else None,
        "hebrew_word_pct": round(100 * heb / n, 1) if n else None,
        "latin_word_pct": round(100 * lat / n, 1) if n else None,
        "empty_junk_segments": junk_segs,
        "trigram_repetition": rep,
        "wall_seconds": getattr(transcript, "_wall_seconds", None),
        "realtime_factor": (round(audio_seconds / transcript._wall_seconds, 2)
                            if audio_seconds and getattr(transcript, "_wall_seconds", 0) else None),
    }


def run_cli(backend_name: str, transcribe_fn):
    import argparse
    ap = argparse.ArgumentParser(description=f"ASR backend: {backend_name}")
    ap.add_argument("video")
    ap.add_argument("--language", default="he")
    ap.add_argument("--out", help="transcript JSON (default exports/audio/<sortie>/transcript.<name>.json)")
    args = ap.parse_args()

    t = timed(transcribe_fn)(args.video, language=args.language or None)
    out = Path(args.out) if args.out else \
        AUDIO_DIR / t.sortie / f"transcript.{backend_name}.json"
    t.save(out)
    import json
    print(f"[{backend_name}] " + json.dumps(metrics(t), ensure_ascii=False))
    print(f"-> {out}")
