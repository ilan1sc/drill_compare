"""Diarization backend: WeSpeaker (CAM++ / ResNet speaker embeddings +
spectral clustering; strong on noisy / far-field audio). Ungated.

Install:
    pip install git+https://github.com/wenet-e2e/wespeaker.git
    # or: pip install wespeaker    (if a release wheel is available)

WeSpeaker ships a one-call diarizer:
    import wespeaker
    m = wespeaker.load_model("english")     # or a local model dir
    m.set_device("cuda")
    segs = m.diarize("audio.wav")           # -> [[start, end, spk_int], ...]

We pass that straight into diarize_common.assign_speakers.

    diarize(transcript, wav_path) -> Transcript
    python3 code/speech/diarize_wespeaker.py exports/audio/<sortie>/transcript.json
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")   # umap/pynndescent set this late otherwise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarize_common import ensure_wav16k, assign_speakers, run_cli  # noqa: E402

MODEL = os.environ.get("WESPEAKER_MODEL", "english")   # name or local dir
_MODEL = None


def _shim_torchaudio():
    """WeSpeaker (+ its s3prl dep) targets an older torchaudio API; on
    torchaudio >= 2.9 these are gone. Restore just enough to import + load."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    if not hasattr(torchaudio, "sox_effects"):
        m = types.ModuleType("torchaudio.sox_effects")
        m.apply_effects_tensor = lambda t, sr, e, **k: (t, sr)
        m.init_sox_effects = m.shutdown_sox_effects = lambda *a, **k: None
        torchaudio.sox_effects = m
        sys.modules["torchaudio.sox_effects"] = m
    _orig_load = torchaudio.load

    def _load(path, normalize=True, **kw):
        try:
            return _orig_load(path, **kw)
        except Exception:
            y, sr = sf.read(str(path), dtype="float32" if normalize else "int16")
            if y.ndim == 1:
                y = y[None, :]
            else:
                y = y.T
            return torch.from_numpy(np.ascontiguousarray(y)), sr
    torchaudio.load = _load

    # wespeaker's UMAP clusterer calls sklearn.check_array(force_all_finite=...),
    # renamed to ensure_all_finite in sklearn >= 1.6
    import sklearn.utils.validation as _skv
    if "force_all_finite" not in _skv.check_array.__code__.co_varnames:
        _orig_ca = _skv.check_array

        def _check_array(*a, **k):
            if "force_all_finite" in k:
                k["ensure_all_finite"] = k.pop("force_all_finite")
            return _orig_ca(*a, **k)
        _skv.check_array = _check_array
        import umap.umap_ as _um
        _um.check_array = _check_array


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        _shim_torchaudio()
        import wespeaker
        m = (wespeaker.load_model_local(MODEL) if os.path.isdir(MODEL)
             else wespeaker.load_model(MODEL))
        m.set_device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = m
    return _MODEL


def diarize(transcript, wav_path, n_speakers=None):
    wav = ensure_wav16k(wav_path)
    m = _model()
    raw = m.diarize(str(wav))          # rows: [utt, start, end, spk] or [start, end, spk]
    segs = []
    for r in raw:
        r = list(r)
        a, b, spk = (r[-3], r[-2], r[-1])
        segs.append((float(a), float(b), str(spk)))
    return assign_speakers(transcript, segs)


if __name__ == "__main__":
    run_cli("wespeaker", diarize)
