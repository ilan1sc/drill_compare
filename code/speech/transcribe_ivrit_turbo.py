"""ASR backend: ivrit-ai/whisper-large-v3-turbo -- the ivrit-ai Hebrew
fine-tune on the Whisper large-v3-turbo backbone (4 decoder layers
instead of 32): markedly faster inference, small accuracy drop vs
ivrit-ai/whisper-large-v3. Good for bulk / first-pass transcription.

Install: transformers (present). Ungated, ~1.6 GB on first use.

    transcribe(video, language="he") -> schema.Transcript
    python3 code/speech/transcribe_ivrit_turbo.py "/path/s1 c2.mp4"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_common import hf_whisper, run_cli  # noqa: E402

MODEL_ID = "ivrit-ai/whisper-large-v3-turbo"


def transcribe(video, language: str | None = "he"):
    return hf_whisper(video, MODEL_ID, language)


if __name__ == "__main__":
    run_cli("ivrit_turbo", transcribe)
