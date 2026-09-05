"""Diarization backend: SpeechBrain ECAPA-TDNN embeddings + spectral
clustering (no pretrained end-to-end pipeline -- this is the embed + cluster
recipe). Ungated (Apache-2.0), model ~80 MB.

Install:
    pip install speechbrain

Method: slide a --win (1.5s) / --hop (0.75s) window over the audio,
restricted to spans the transcript actually has words in (cheap VAD),
embed each with speechbrain/spkrec-ecapa-voxceleb, build a cosine-similarity
affinity, SpectralClustering into 2, emit per-window (start,end,label).
diarize_common.assign_speakers then maps those onto words with smoothing.

    diarize(transcript, wav_path) -> Transcript
    python3 code/speech/diarize_speechbrain.py exports/audio/<sortie>/transcript.json
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarize_common import ensure_wav16k, assign_speakers, run_cli  # noqa: E402

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        _ENC = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "speechbrain" / "ecapa"),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    return _ENC


def _speech_spans(transcript, pad=0.2):
    spans, cur = [], None
    words = sorted((w for s in transcript.segments for w in s.words), key=lambda w: w.start)
    for w in words:
        if cur and w.start - cur[1] <= 0.6:
            cur[1] = max(cur[1], w.end)
        else:
            if cur:
                spans.append(cur)
            cur = [w.start, w.end]
    if cur:
        spans.append(cur)
    return [(max(0.0, a - pad), b + pad) for a, b in spans]


def diarize(transcript, wav_path, n_speakers: int = 2, win: float = 1.5, hop: float = 0.75):
    import soundfile as sf
    import torch
    from sklearn.cluster import SpectralClustering

    wav = ensure_wav16k(wav_path)
    y, sr = sf.read(str(wav))
    if y.ndim > 1:
        y = y.mean(1)

    enc = _encoder()
    wins, embs = [], []
    for a, b in _speech_spans(transcript):
        t = a
        while t + win <= b + 1e-6 or (not wins and t < b):
            i0, i1 = int(t * sr), int(min(t + win, b) * sr)
            if i1 - i0 >= int(0.4 * sr):
                seg = torch.tensor(y[i0:i1], dtype=torch.float32).unsqueeze(0)
                e = enc.encode_batch(seg).squeeze().detach().cpu().numpy()
                embs.append(e / (np.linalg.norm(e) + 1e-9))
                wins.append((t, min(t + win, b)))
            t += hop

    if len(embs) < n_speakers:
        return assign_speakers(transcript, [(w[0], w[1], "0") for w in wins])

    X = np.vstack(embs)
    aff = np.clip((X @ X.T + 1.0) / 2.0, 0, 1)     # cosine -> [0,1] affinity
    labels = SpectralClustering(n_clusters=n_speakers, affinity="precomputed",
                                assign_labels="discretize", random_state=0).fit_predict(aff)
    segs = [(wins[i][0], wins[i][1], str(int(labels[i]))) for i in range(len(wins))]
    return assign_speakers(transcript, segs)


if __name__ == "__main__":
    run_cli("speechbrain-ecapa", diarize)
