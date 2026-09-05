"""ASR backend: ivrit-ai/yi-whisper-large-v3 -- ivrit-ai's later Hebrew
fine-tune of Whisper large-v3 (trained on a larger / cleaner Hebrew set
than ivrit-ai/whisper-large-v3). Same interface, swap to compare accuracy.

Install: transformers (present). Ungated, ~3 GB on first use.

    transcribe(video, language="he") -> schema.Transcript
    python3 code/speech/transcribe_ivrit_yi.py "/path/s1 c2.mp4"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_common import hf_whisper, run_cli  # noqa: E402

MODEL_ID = "ivrit-ai/yi-whisper-large-v3"


def transcribe(video, language: str | None = "he"):
    return hf_whisper(video, MODEL_ID, language)


if __name__ == "__main__":
    run_cli("ivrit_yi", transcribe)
