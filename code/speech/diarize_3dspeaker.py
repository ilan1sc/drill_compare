"""Diarization backend: 3D-Speaker (Alibaba / ModelScope) -- CAM++ based
speaker-diarization pipeline. Ungated; models pull from ModelScope.

Install:
    pip install modelscope
    pip install "speakerlab @ git+https://github.com/modelscope/3D-Speaker.git"

Two APIs exist depending on version; this tries the modelscope pipeline
first, then the speakerlab Diarization3Dspeaker class:

    from modelscope.pipelines import pipeline
    p = pipeline(task='speaker-diarization',
                 model='damo/speech_campplus_speaker-diarization_common')
    out = p(wav_path, oracle_num=2)      # -> {'text': [[start, end, spk], ...]}

    diarize(transcript, wav_path) -> Transcript
    python3 code/speech/diarize_3dspeaker.py exports/audio/<sortie>/transcript.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarize_common import ensure_wav16k, assign_speakers, run_cli  # noqa: E402

MODEL_ID = "damo/speech_campplus_speaker-diarization_common"
_PIPE = None


def _segments_modelscope(wav: str, n_speakers):
    global _PIPE
    if _PIPE is None:
        from modelscope.pipelines import pipeline
        _PIPE = pipeline(task="speaker-diarization", model=MODEL_ID)
    out = _PIPE(str(wav), **({"oracle_num": int(n_speakers)} if n_speakers else {}))
    rows = out["text"] if isinstance(out, dict) else out
    return [(float(r[0]), float(r[1]), str(r[2])) for r in rows]


def _segments_speakerlab(wav: str, n_speakers):
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
    d = Diarization3Dspeaker(device="cuda", include_overlap=True)
    res = d(str(wav), **({"num_speakers": int(n_speakers)} if n_speakers else {}))
    return [(float(r[0]), float(r[1]), str(r[2])) for r in res]


def diarize(transcript, wav_path, n_speakers=None):
    """n_speakers: int forces exactly that many; None lets 3D-Speaker's own
    spectral clustering pick the count (recommended when it's uncertain --
    e.g. crew + sim instructor + HE/EN sim callouts, up to ~5)."""
    wav = ensure_wav16k(wav_path)
    try:
        segs = _segments_modelscope(str(wav), n_speakers)
    except Exception:
        segs = _segments_speakerlab(str(wav), n_speakers)
    return assign_speakers(transcript, segs)


if __name__ == "__main__":
    run_cli("3d-speaker", diarize)
