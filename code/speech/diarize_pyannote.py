"""Diarization backend: pyannote.audio 3.1 (the reference pipeline --
segmentation + speaker embedding + clustering + overlap handling).

Install:
    pip install pyannote.audio
Gating (one-time, free):
    1. accept the conditions on https://huggingface.co/pyannote/speaker-diarization-3.1
       and https://huggingface.co/pyannote/segmentation-3.0
    2. create a token at https://huggingface.co/settings/tokens
    3. export HF_TOKEN=hf_xxx   (or pass --hf-token / set HUGGINGFACE_TOKEN)

We tell it num_speakers=2 (cadet + instructor is known).

    diarize(transcript, wav_path) -> Transcript
    HF_TOKEN=hf_xxx python3 code/speech/diarize_pyannote.py exports/audio/<sortie>/transcript.json
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarize_common import ensure_wav16k, assign_speakers, run_cli  # noqa: E402

MODEL_ID = "pyannote/speaker-diarization-3.1"
_PIPE = None


def _pipeline():
    global _PIPE
    if _PIPE is None:
        import torch
        from pyannote.audio import Pipeline
        token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if not token:
            raise RuntimeError(
                "pyannote models are gated: accept the license on huggingface.co "
                "for pyannote/speaker-diarization-3.1 + pyannote/segmentation-3.0, "
                "then set HF_TOKEN. See this module's docstring.")
        pipe = Pipeline.from_pretrained(MODEL_ID, use_auth_token=token)
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
        _PIPE = pipe
    return _PIPE


def _segments(wav: str, n_speakers: int = 2):
    diarization = _pipeline()(str(wav), num_speakers=n_speakers)
    return [(turn.start, turn.end, label)
            for turn, _, label in diarization.itertracks(yield_label=True)]


def diarize(transcript, wav_path, n_speakers: int = 2):
    wav = ensure_wav16k(wav_path)
    return assign_speakers(transcript, _segments(str(wav), n_speakers))


if __name__ == "__main__":
    run_cli("pyannote-3.1", diarize)
