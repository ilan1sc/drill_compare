"""ASR backend: Meta MMS (facebook/mms-1b-all) -- a 1B-param wav2vec2
CTC model with per-language adapters (Hebrew adapter id 'heb'). CTC
models don't hallucinate the way autoregressive Whisper does (no decoder
inventing text over silence/noise), so this is the hallucination-resistant
option -- at some cost in punctuation/casing and fluency.

Install: transformers (present). Ungated, ~4 GB on first use (base model
+ the 'heb' adapter).

No per-word confidence from CTC via the HF pipeline, so Word.prob is left
None and the report's confidence columns are blank for this backend.

    transcribe(video, language="heb") -> schema.Transcript
    python3 code/speech/transcribe_mms.py "/path/s1 c2.mp4"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_common import build_transcript, load_wav, run_cli  # noqa: E402
from schema import Word  # noqa: E402

MODEL_ID = "facebook/mms-1b-all"
_PIPE = None


def _pipe(lang_code: str):
    global _PIPE
    if _PIPE is None:
        import torch
        from transformers import pipeline
        _PIPE = pipeline("automatic-speech-recognition", model=MODEL_ID,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    _PIPE.tokenizer.set_target_lang(lang_code)
    _PIPE.model.load_adapter(lang_code)
    return _PIPE


def transcribe(video, language: str | None = "heb"):
    lang = language or "heb"
    if lang == "he":
        lang = "heb"                       # MMS uses ISO-639-3 adapter ids
    y, sr, wav = load_wav(video)
    out = _pipe(lang)({"array": y, "sampling_rate": sr}, return_timestamps="word",
                      chunk_length_s=20, stride_length_s=(4, 2))
    words = [Word(text=c["text"].strip(),
                 start=float(c["timestamp"][0] or 0.0),
                 end=float(c["timestamp"][1] or c["timestamp"][0] or 0.0), prob=None)
            for c in out.get("chunks", []) if c["text"].strip()]
    return build_transcript(Path(video).stem, video, words)


if __name__ == "__main__":
    run_cli("mms", transcribe)
