"""Diarization backend: NVIDIA NeMo Sortformer (end-to-end neural, handles
overlapped speech). Model: nvidia/diar_sortformer_4spk-v1 (4-speaker
capacity; the two most-talkative predicted speakers become SPEAKER_00/01).

Install:  pip install "nemo_toolkit[asr]"        (no HF gating; model
          auto-downloads to ~/.cache/huggingface on first use)
Tested working on this box (aarch64 + CUDA), ~15s model load, ~1s per
minute of audio on GPU.

    diarize(transcript, wav_path) -> Transcript
    python3 code/speech/diarize_nemo.py exports/audio/<sortie>/transcript.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarize_common import ensure_wav16k, assign_speakers, run_cli  # noqa: E402

MODEL_ID = "nvidia/diar_sortformer_4spk-v1"
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel
        m = SortformerEncLabelModel.from_pretrained(MODEL_ID)
        m.eval()
        if torch.cuda.is_available():
            m = m.to("cuda")
        _MODEL = m
    return _MODEL


def _segments(wav: str):
    out = _model().diarize(audio=[str(wav)], batch_size=1)
    raw = out[0] if out and isinstance(out, (list, tuple)) else out
    segs = []
    for item in raw:
        if isinstance(item, str):
            a, b, spk = item.split()
        else:
            a, b, spk = item[0], item[1], item[2]
        segs.append((float(a), float(b), str(spk)))
    return segs


def diarize(transcript, wav_path, n_speakers=None):   # n_speakers kept for API parity
    wav = ensure_wav16k(wav_path)
    return assign_speakers(transcript, _segments(str(wav)))


if __name__ == "__main__":
    run_cli("nemo-sortformer", diarize)
