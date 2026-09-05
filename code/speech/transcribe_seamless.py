"""ASR backend: Meta SeamlessM4T (v2 Large by default, or v1 Medium via
--variant medium). Multimodal speech/text model with strong handling of
code-switched speech -- the best of this set at mixed Hebrew/English
(e.g. Hebrew crew talk with English sim callouts in the same utterance).

Install: transformers (present). Ungated. v2-large ~9 GB, v1-medium ~4 GB
on first use.

SeamlessM4T is sequence-to-sequence with NO timestamps of any kind, so
this backend cuts fixed --win-second windows itself and spreads each
window's words evenly across its span. Word.prob is None. Subtitle timing
is therefore approximate (window-level), fine for the comparison movie
and the word-count report, not for tight sync.

    transcribe(video, language="heb", variant="large") -> schema.Transcript
    python3 code/speech/transcribe_seamless.py "/path/s1 c2.mp4" --variant large
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_common import build_transcript, chunk, load_wav  # noqa: E402
from schema import Segment, Word  # noqa: E402

VARIANTS = {"large": ("facebook/seamless-m4t-v2-large", "SeamlessM4Tv2Model"),
            "medium": ("facebook/seamless-m4t-medium", "SeamlessM4TModel")}
_CACHE: dict = {}


def _load(variant: str):
    if variant not in _CACHE:
        import torch
        import transformers
        from transformers import AutoProcessor
        model_id, cls_name = VARIANTS[variant]
        cls = getattr(transformers, cls_name)
        proc = AutoProcessor.from_pretrained(model_id)
        model = cls.from_pretrained(model_id)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _CACHE[variant] = (proc, model)
    return _CACHE[variant]


def transcribe(video, language: str | None = "heb", variant: str = "large", win_s: float = 20.0):
    import torch
    lang = "heb" if (language in (None, "he", "heb")) else language
    proc, model = _load(variant)
    y, sr, wav = load_wav(video)

    words, segs = [], []
    for t0, seg in chunk(y, sr, win_s=win_s, hop_s=win_s):
        try:
            inputs = proc(audio=seg, sampling_rate=sr, return_tensors="pt").to(model.device)
        except TypeError:
            inputs = proc(audios=seg, sampling_rate=sr, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, tgt_lang=lang, generate_speech=False)
        # v2 returns a namedtuple/obj with .sequences; v1 returns a tensor;
        # some builds return a nested list. Normalise to a flat id list.
        ids = getattr(out, "sequences", out)
        try:
            ids = ids[0].tolist()
        except Exception:
            ids = list(ids)
        while isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        text = proc.decode(ids, skip_special_tokens=True).strip()
        if not text:
            continue
        parts = text.split()
        dur = min(win_s, len(seg) / sr)
        step = dur / max(len(parts), 1)
        ws = [Word(text=p, start=round(t0 + k * step, 2),
                  end=round(t0 + (k + 1) * step, 2), prob=None)
             for k, p in enumerate(parts)]
        words += ws
        segs.append(Segment(id=len(segs), start=round(t0, 2), end=round(t0 + dur, 2),
                            text=text, lang="he", speaker=None, words=ws))
    return build_transcript(Path(video).stem, video, words, segments=segs)


if __name__ == "__main__":
    import argparse
    from transcribe_common import AUDIO_DIR, metrics, timed
    import json

    ap = argparse.ArgumentParser(description="ASR backend: SeamlessM4T")
    ap.add_argument("video")
    ap.add_argument("--language", default="heb")
    ap.add_argument("--variant", choices=list(VARIANTS), default="large")
    ap.add_argument("--out")
    a = ap.parse_args()

    t = timed(transcribe)(a.video, language=a.language, variant=a.variant)
    out = Path(a.out) if a.out else AUDIO_DIR / t.sortie / f"transcript.seamless_{a.variant}.json"
    t.save(out)
    print(f"[seamless_{a.variant}] " + json.dumps(metrics(t), ensure_ascii=False))
    print(f"-> {out}")
