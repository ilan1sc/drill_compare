"""Diarization backend: WhisperX's diarization step
(whisperx.diarize.DiarizationPipeline). NOTE: this is pyannote under the
hood, so it has the SAME gating as diarize_pyannote.py -- accept the
pyannote licenses + set HF_TOKEN. Included as its own backend because
WhisperX pins/handles the pyannote version itself and is what a lot of
existing WhisperX pipelines already use.

Install:
    pip install whisperx
Gating: same as diarize_pyannote.py (HF_TOKEN + accept pyannote licenses).

    diarize(transcript, wav_path) -> Transcript
    HF_TOKEN=hf_xxx python3 code/speech/diarize_whisperx.py exports/audio/<sortie>/transcript.json
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarize_common import ensure_wav16k, assign_speakers, run_cli  # noqa: E402

_PIPE = None


def _pipeline():
    global _PIPE
    if _PIPE is None:
        import torch
        import whisperx
        token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if not token:
            raise RuntimeError("WhisperX diarization wraps pyannote -- set HF_TOKEN "
                               "and accept the pyannote licenses (see docstring).")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _PIPE = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=device)
        except AttributeError:                       # older whisperx layout
            from whisperx import DiarizationPipeline
            _PIPE = DiarizationPipeline(use_auth_token=token, device=device)
    return _PIPE


def diarize(transcript, wav_path, n_speakers: int = 2):
    wav = ensure_wav16k(wav_path)
    df = _pipeline()(str(wav), min_speakers=n_speakers, max_speakers=n_speakers)
    # df is a pandas DataFrame with columns start / end / speaker
    segs = [(float(r["start"]), float(r["end"]), str(r["speaker"]))
            for _, r in df.iterrows()]
    return assign_speakers(transcript, segs)


if __name__ == "__main__":
    run_cli("whisperx", diarize)
