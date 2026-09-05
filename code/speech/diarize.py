"""Lightweight speaker diarization: no external account/token needed.

Approach (good enough to separate cadet vs instructor on a 2-person
cockpit channel, not broadcast-quality):
  1. Re-split whisper's segments at internal word gaps > --pause-gap so
     each chunk is (close to) one speaker turn.
  2. MFCC-statistics embedding (mean + std of 20 MFCCs) per chunk via
     librosa, straight off the mono 16kHz wav already extracted for ASR.
  3. Agglomerative clustering into --speakers clusters.
  4. Label clusters SPEAKER_00.. in order of first appearance, write the
     label onto every Segment/Word.

For real diarization quality, swap this module for pyannote.audio (needs
a HuggingFace token + accepting the pyannote/speaker-diarization license)
-- pipeline.py only expects `diarize(transcript, wav_path) -> Transcript`.

    python3 code/speech/diarize.py exports/audio/<sortie>/transcript.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Transcript  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"


def _resplit(transcript: Transcript, pause_gap: float):
    """yield (start, end, word_indices) chunks split at long pauses."""
    for seg in transcript.segments:
        if not seg.words:
            yield seg, [], seg.start, seg.end
            continue
        run = [0]
        for i in range(1, len(seg.words)):
            if seg.words[i].start - seg.words[i - 1].end > pause_gap:
                yield seg, run, seg.words[run[0]].start, seg.words[run[-1]].end
                run = []
            run.append(i)
        if run:
            yield seg, run, seg.words[run[0]].start, seg.words[run[-1]].end


def _embed(y, sr, t0, t1):
    import librosa
    i0, i1 = int(t0 * sr), max(int(t0 * sr) + 1, int(t1 * sr))
    clip = y[i0:i1]
    if len(clip) < sr * 0.15:                 # too short to trust
        return None
    mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def diarize(transcript: Transcript, wav_path: str | Path,
           n_speakers: int = 5, pause_gap: float = 0.6) -> Transcript:
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)

    chunks = list(_resplit(transcript, pause_gap))
    feats, idx = [], []
    for k, (seg, word_idx, t0, t1) in enumerate(chunks):
        e = _embed(y, sr, t0, t1)
        if e is not None:
            feats.append(e)
            idx.append(k)

    if len(feats) < n_speakers:
        for seg, word_idx, t0, t1 in chunks:
            seg.speaker = "SPEAKER_00"
            for i in word_idx:
                seg.words[i].speaker = "SPEAKER_00"
        return transcript

    X = StandardScaler().fit_transform(np.array(feats))
    labels = AgglomerativeClustering(n_clusters=n_speakers).fit_predict(X)

    # relabel clusters in order of first appearance -> SPEAKER_00, 01, ...
    order, seen = {}, 0
    for lab in labels:
        if lab not in order:
            order[lab] = f"SPEAKER_{seen:02d}"
            seen += 1

    label_by_chunk = {idx[i]: order[lab] for i, lab in enumerate(labels)}
    for k, (seg, word_idx, t0, t1) in enumerate(chunks):
        spk = label_by_chunk.get(k, "SPEAKER_UNK")
        for i in word_idx:
            seg.words[i].speaker = spk
    # segment-level speaker = majority vote of its words
    for seg in transcript.segments:
        if seg.words:
            vals = [w.speaker for w in seg.words if w.speaker]
            seg.speaker = max(set(vals), key=vals.count) if vals else "SPEAKER_UNK"

    return transcript


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript_json")
    ap.add_argument("--wav", help="wav path (default exports/audio/<sortie>.wav)")
    ap.add_argument("--speakers", type=int, default=5)
    ap.add_argument("--pause-gap", type=float, default=0.6)
    args = ap.parse_args()

    t = Transcript.load(args.transcript_json)
    wav = Path(args.wav) if args.wav else AUDIO_DIR / f"{t.sortie}.wav"
    t = diarize(t, wav, n_speakers=args.speakers, pause_gap=args.pause_gap)
    t.save(args.transcript_json)
    counts = {}
    for w in t.words():
        counts[w.speaker] = counts.get(w.speaker, 0) + 1
    print(f"speakers: {counts} -> {args.transcript_json}")


if __name__ == "__main__":
    main()
