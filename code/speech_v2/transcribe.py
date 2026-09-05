"""SOLUTION 2 -- speech-to-text with a Hebrew-specialized ASR model.

code/speech/transcribe.py (solution 1) uses openai-whisper large-v3,
a general-purpose multilingual model. Almost all of the cockpit audio
in this project is Hebrew, so solution 2 swaps in
`ivrit-ai/whisper-large-v3` (https://huggingface.co/ivrit-ai/whisper-large-v3)
-- a large-v3 checkpoint fine-tuned on a large Hebrew speech corpus.
On a spot check (same 35s clip, both solutions) it produced noticeably
more coherent Hebrew ("זה ביני לבין המדריכת" vs solution 1's
"זה פני לבין המדריכת") and fewer garbled radio-callout words
("בשדר גמור" consistently vs "שר גמור" / "בסר גמור").

Trade-off to know about: --language defaults to "he" (forced), which is
what makes the Hebrew gains possible, but it means occasional English
(PFD read-backs, aviation abbreviations the crew says in English) gets
transcribed as Hebrew-phonetic noise instead of being recognized as
English the way solution 1's free language auto-detect could. Pass
--language "" (empty) for auto-detect if a sortie is more English-heavy.

Loaded via transformers (not openai-whisper, which can only load OpenAI's
own checkpoint format) as float16 on CUDA. Output is the same
code/speech/schema.py Transcript shape solution 1 uses, so every existing
diarize.py / glossary.py / biomarkers module works unchanged on it -- only
the ASR step and the output tree (exports/audio_v2/) are new.

    python3 code/speech_v2/transcribe.py "/path/s1 c2.mp4"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speech"))
from extract_audio import extract  # noqa: E402
from schema import Segment, Transcript, Word  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio_v2"
MODEL_ID = "ivrit-ai/whisper-large-v3"

_PIPE = None


def _load_pipe():
    global _PIPE
    if _PIPE is None:
        import torch
        from transformers import pipeline
        _PIPE = pipeline(
            "automatic-speech-recognition", model=MODEL_ID,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _PIPE


def _words_to_segments(chunks, pause_gap: float = 0.6) -> list[Segment]:
    """The HF pipeline returns a flat per-word list (no segment grouping
    like openai-whisper does) -- rebuild segments by splitting at pauses,
    same convention diarize.py uses for its own re-split."""
    words = [
        Word(text=c["text"].strip(), start=float(c["timestamp"][0]),
            end=float(c["timestamp"][1] or c["timestamp"][0]), prob=1.0)
        for c in chunks if c["text"].strip()
    ]
    segments, cur = [], []
    for w in words:
        if cur and w.start - cur[-1].end > pause_gap:
            segments.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segments.append(cur)

    out = []
    for i, ws in enumerate(segments):
        out.append(Segment(id=i, start=ws[0].start, end=ws[-1].end,
                           text=" ".join(w.text for w in ws), lang="he",
                           speaker=None, words=ws))
    return out


def transcribe(video: str | Path, language: str = "he") -> Transcript:
    import librosa

    video = Path(video)
    wav = extract(video, out_dir=AUDIO_DIR)
    y, sr = librosa.load(str(wav), sr=16000, mono=True)

    pipe = _load_pipe()
    gen_kwargs = {"task": "transcribe"}
    if language:
        gen_kwargs["language"] = language
    t0 = time.time()
    result = pipe({"array": np.asarray(y), "sampling_rate": 16000},
                  return_timestamps="word", chunk_length_s=28, stride_length_s=4,
                  generate_kwargs=gen_kwargs)
    elapsed = time.time() - t0

    segments = _words_to_segments(result.get("chunks", []))
    duration = segments[-1].end if segments else 0.0
    print(f"  [{MODEL_ID}] {len(segments)} segments, "
          f"{sum(len(s.words) for s in segments)} words, {duration:.0f}s  ({elapsed:.0f}s)")

    return Transcript(sortie=video.stem, source_video=str(video),
                      duration=duration, segments=segments)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--language", default="he", help='force this language, or "" to auto-detect')
    ap.add_argument("--out", help="transcript JSON path (default exports/audio_v2/<sortie>/transcript.json)")
    args = ap.parse_args()

    t = transcribe(args.video, language=args.language or None)
    out = Path(args.out) if args.out else AUDIO_DIR / t.sortie / "transcript.json"
    t.save(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
