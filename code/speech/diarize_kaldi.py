"""Diarization backend: Kaldi's x-vector + PLDA + agglomerative-clustering
recipe (the callhome_diarization / DIHARD pipeline).

Kaldi is NOT pip-installable -- it's a C++ toolkit you compile. This
module shells out to an existing Kaldi tree and a pretrained x-vector
diarization model:

    export KALDI_ROOT=/opt/kaldi
    python3 code/speech/diarize_kaldi.py exports/audio/<sortie>/transcript.json \\
        --xvector-model /opt/kaldi/egs/callhome_diarization/v2/exp/xvector_nnet_1a

    (--xvector-model dir must contain: final.raw, extract.config, plda,
     mean.vec, transform.mat -- e.g. the released 0008_sitw_v2_1a model.)

Steps run (standard Kaldi scripts, from $KALDI_ROOT/egs/callhome_diarization/v2):
    make MFCC (conf/mfcc.conf, 23-dim) -> sliding-window CMVN
    steps/nnet3/xvector/extract_xvectors.sh   (--window 1.5 --period 0.75)
    diarization/nnet3/xvector/score_plda.sh
    diarization/cluster.sh --threshold <tuned>   (or --reco2num-spk for oracle 2)
    -> RTTM, parsed here into (start, end, speaker) for
       diarize_common.assign_speakers.

    diarize(transcript, wav_path) -> Transcript
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarize_common import ensure_wav16k, assign_speakers, run_cli  # noqa: E402

XVEC_MODEL = os.environ.get("KALDI_XVECTOR_MODEL", "")


def _kaldi_root() -> Path:
    root = os.environ.get("KALDI_ROOT")
    if not root or not Path(root).is_dir():
        raise RuntimeError("Kaldi not found: compile Kaldi and `export KALDI_ROOT=/path/to/kaldi`. "
                           "Kaldi has no pip package -- see this module's docstring.")
    return Path(root)


def _parse_rttm(rttm_path: Path):
    segs = []
    for line in Path(rttm_path).read_text().splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            segs.append((start, start + dur, spk))
    return segs


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def diarize(transcript, wav_path, n_speakers: int = 2, xvector_model: str | None = None):
    kaldi = _kaldi_root()
    v2 = kaldi / "egs" / "callhome_diarization" / "v2"
    model = Path(xvector_model or XVEC_MODEL)
    if not model.is_dir():
        raise RuntimeError("--xvector-model / $KALDI_XVECTOR_MODEL must point at a trained "
                           "x-vector dir (final.raw, extract.config, plda, mean.vec, transform.mat).")

    wav = ensure_wav16k(wav_path)
    env = {**os.environ, "KALDI_ROOT": str(kaldi),
           "PATH": f"{kaldi/'src'/'featbin'}:{kaldi/'src'/'bin'}:"
                   f"{kaldi/'src'/'ivectorbin'}:{kaldi/'src'/'nnet3bin'}:{os.environ['PATH']}"}

    work = Path(tempfile.mkdtemp(prefix="kaldi_diar_"))
    try:
        data = work / "data"
        data.mkdir(parents=True)
        rec = transcript.sortie.replace(" ", "_")
        (data / "wav.scp").write_text(f"{rec} {wav}\n")
        (data / "segments").write_text(f"{rec}-0 {rec} 0.00 {transcript.duration:.2f}\n")
        (data / "utt2spk").write_text(f"{rec}-0 {rec}\n")
        (data / "spk2utt").write_text(f"{rec} {rec}-0\n")

        mfcc = work / "mfcc"
        _run(["steps/make_mfcc.sh", "--mfcc-config", "conf/mfcc.conf", "--nj", "1",
              "--cmd", "run.pl", str(data), str(work / "log_mfcc"), str(mfcc)], v2, env)
        _run(["local/nnet3/xvector/prepare_feats.sh", "--nj", "1", "--cmd", "run.pl",
              str(data), str(work / "data_cmn"), str(work / "cmn")], v2, env)

        xvec = work / "xvectors"
        _run(["diarization/nnet3/xvector/extract_xvectors.sh", "--cmd", "run.pl", "--nj", "1",
              "--window", "1.5", "--period", "0.75", "--apply-cmn", "false",
              "--min-segment", "0.5", str(model), str(work / "data_cmn"), str(xvec)], v2, env)

        scores = work / "scores"
        _run(["diarization/nnet3/xvector/score_plda.sh", "--cmd", "run.pl", "--target-energy", "0.9",
              "--nj", "1", str(model), str(xvec), str(scores)], v2, env)

        clus = work / "cluster"
        reco2num = work / "reco2num_spk"
        reco2num.write_text(f"{rec} {n_speakers}\n")
        _run(["diarization/cluster.sh", "--cmd", "run.pl", "--nj", "1",
              "--reco2num-spk", str(reco2num), str(scores), str(clus)], v2, env)

        return assign_speakers(transcript, _parse_rttm(clus / "rttm"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import Transcript
    from diarize_common import AUDIO_DIR, print_summary

    ap = argparse.ArgumentParser(description="Diarization backend: kaldi x-vector + PLDA + AHC")
    ap.add_argument("transcript_json")
    ap.add_argument("--wav")
    ap.add_argument("--xvector-model", default=None)
    a = ap.parse_args()
    t = Transcript.load(a.transcript_json)
    wav = Path(a.wav) if a.wav else AUDIO_DIR / f"{t.sortie}.wav"
    t = diarize(t, wav, xvector_model=a.xvector_model)
    t.save(a.transcript_json)
    print_summary(t, a.transcript_json)
