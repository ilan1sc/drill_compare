"""ASR backend: ivrit-ai/whisper-large-v3 -- Whisper large-v3 fine-tuned
on extensive Hebrew corpora. State-of-the-art Hebrew accuracy; this is
what code/speech_v2/pipeline.py (solution 2) already uses.

Install: transformers (already present). Ungated, ~3 GB on first use.
Trade-off: --language defaults to 'he' (forced) for the Hebrew gains;
English gets transcribed as Hebrew-phonetic noise. Pass --language "" to
auto-detect.

    transcribe(video, language="he") -> schema.Transcript
    python3 code/speech/transcribe_ivrit_v3.py "/path/s1 c2.mp4"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_common import hf_whisper, run_cli  # noqa: E402

MODEL_ID = "ivrit-ai/whisper-large-v3"


def transcribe(video, language: str | None = "he"):
    return hf_whisper(video, MODEL_ID, language)


if __name__ == "__main__":
    run_cli("ivrit_v3", transcribe)
