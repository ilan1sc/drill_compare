"""Language-ID refinement pass -- splits out the simulator's spoken
callouts (which come in Hebrew AND English) as their own speaker classes,
which the acoustic diarizers alone tend to merge into the crew.

Runs AFTER any speaker-diarization backend. For every segment it detects
the spoken language from that segment's audio (openai-whisper --model,
default 'base', per-segment `detect_language`). Segments whose top
language isn't Hebrew, with probability >= --min-prob, are relabelled:

    <lang> == en      -> SIM_EN     (English simulator callout)
    <lang> != he, en  -> SIM_<LANG> (e.g. SIM_RU) -- rare, flagged for review

Hebrew segments keep whatever speaker label the acoustic backend gave
them (crew / sim instructor). This does NOT try to separate the Hebrew
sim callout voice from the humans -- that needs the acoustic backend at
--speakers >= 4; langid only reliably isolates the English one.

CAVEAT (measured on s1 c2): per-segment whisper language detection is
UNRELIABLE on short (< ~3s), noisy intercom clips -- it will confidently
mislabel Hebrew speech as random languages. Defaults are therefore
conservative (--model large-v3, --min-prob 0.85, --min-dur 3.0) and this
pass should be treated as a candidate list to eyeball, not ground truth.
Raising --speakers on 3dspeaker / pyannote is the more trustworthy way to
surface the extra sim voices.

    python3 code/speech/diarize_langid.py exports/audio/<sortie>/transcript.json --wav ...
    (or import: refine(transcript, wav_path) -> Transcript, after a backend)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarize_common import ensure_wav16k, print_summary, run_cli  # noqa: E402

_MODEL = None


def _model(name: str):
    global _MODEL
    if _MODEL is None:
        import torch
        import whisper
        _MODEL = whisper.load_model(name, device="cuda" if torch.cuda.is_available() else "cpu")
    return _MODEL


def refine(transcript, wav_path, model_name: str = "large-v3", min_prob: float = 0.85,
          min_dur: float = 3.0):
    import soundfile as sf
    import whisper

    wav = ensure_wav16k(wav_path)
    y, sr = sf.read(str(wav))
    if y.ndim > 1:
        y = y.mean(1)
    y = y.astype(np.float32)
    model = _model(model_name)

    changed = 0
    for seg in transcript.segments:
        if seg.end - seg.start < min_dur:
            continue
        clip = y[int(seg.start * sr):int(seg.end * sr)]
        if len(clip) < int(min_dur * sr):
            continue
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(clip), model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        p = probs[lang]
        if lang == "he" or p < min_prob:
            continue
        new = "SIM_EN" if lang == "en" else f"SIM_{lang.upper()}"
        seg.speaker = new
        for w in seg.words:
            w.speaker = new
        changed += 1

    print(f"langid: relabelled {changed} / {len(transcript.segments)} segments as SIM_*")
    return transcript


if __name__ == "__main__":
    run_cli("langid-refine", refine)
