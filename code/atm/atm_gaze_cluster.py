"""Auto-calibrate "is the cadet looking at the PFD" from the face-cam gaze
track, with NO hand-picked timestamps.

Premise (given by the crew): inside the cockpit the cadet's gaze is ~95%
on the PFD -- the engine/MFD screen is not used at this stage. So the
dominant *inside* (head pitched toward the panel) cluster of gaze
directions simply IS the PFD. We:

  1. load exports/gaze/<sortie>/<seat>.json  (cadet = "left" seat in the
     face cam; --seat to override),
  2. split samples into "inside" vs "outside" by pitch (KMeans k=2 on
     pitch; the lower/down centroid = inside),
  3. within "inside", take the densest cluster as the PFD direction
     (centroid + per-axis MAD spread),
  4. label every sample:
        on_pfd        -- within --tol * MAD of the PFD centroid
        other_inside  -- pitched down but off the PFD cluster (~the 5%)
        outside       -- head level / up (windscreen etc.)
  5. write exports/atm/<sortie>/gaze_pfd.json :
        {seat, pfd_centroid:{yaw,pitch}, pfd_mad:{yaw,pitch}, tol,
         hz, on_pfd_pct, other_inside_pct, outside_pct,
         samples:[{t, yaw, pitch, label}]}

    python3 code/atm/atm_gaze_cluster.py --sortie "s1 c6"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
GAZE_DIR = ROOT / "exports" / "gaze"
ATM_DIR = ROOT / "exports" / "atm"
sys.path.insert(0, str(ROOT / "code" / "gaze"))
from schema import GazeTrack  # noqa: E402


def _km1d(x, iters=50):
    """tiny 2-means on a 1-D array -> (labels, centers) with center[0] < center[1]."""
    x = np.asarray(x, float)
    c = np.array([np.percentile(x, 25), np.percentile(x, 75)])
    lab = np.zeros(len(x), int)
    for _ in range(iters):
        lab = (np.abs(x[:, None] - c[None, :]).argmin(1))
        for k in (0, 1):
            if (lab == k).any():
                c[k] = x[lab == k].mean()
    order = np.argsort(c)
    remap = {order[0]: 0, order[1]: 1}
    return np.array([remap[v] for v in lab]), c[order]


def _km2d(pts, k, iters=80, seed=0):
    rng = np.random.default_rng(seed)
    c = pts[rng.choice(len(pts), k, replace=False)]
    lab = np.zeros(len(pts), int)
    for _ in range(iters):
        d = ((pts[:, None, :] - c[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = pts[lab == j].mean(0)
    return lab, c


def cluster(track: GazeTrack, k_inside: int = 3, tol: float = 3.0):
    s = track.samples
    yaw = np.array([g.yaw for g in s], float)
    pitch = np.array([g.pitch for g in s], float)

    # inside vs outside by pitch: the "down" centroid is the panel
    plab, pc = _km1d(pitch)
    inside_mask = plab == 0            # lower pitch centroid
    if inside_mask.sum() < 10:         # degenerate -- fall back to "below median pitch"
        inside_mask = pitch <= np.median(pitch)

    ipts = np.column_stack([yaw[inside_mask], pitch[inside_mask]])
    kk = min(k_inside, max(1, len(ipts) // 20))
    ilab, ic = _km2d(ipts, kk) if kk > 1 else (np.zeros(len(ipts), int), ipts.mean(0, keepdims=True))
    # densest inside cluster = PFD
    counts = np.bincount(ilab, minlength=len(ic))
    pfd_j = int(counts.argmax())
    pfd_pts = ipts[ilab == pfd_j]
    cen = pfd_pts.mean(0)
    mad = np.median(np.abs(pfd_pts - cen), 0)
    mad = np.where(mad < 1e-3, 1.0, mad)

    labels = []
    for i, g in enumerate(s):
        dz = np.abs(np.array([g.yaw, g.pitch]) - cen) / mad
        if not inside_mask[i]:
            labels.append("outside")
        elif (dz <= tol).all():
            labels.append("on_pfd")
        else:
            labels.append("other_inside")

    n = len(labels) or 1
    return {
        "seat": track.seat,
        "hz": track.sample_hz,
        "duration": track.duration,
        "pfd_centroid": {"yaw": round(float(cen[0]), 3), "pitch": round(float(cen[1]), 3)},
        "pfd_mad": {"yaw": round(float(mad[0]), 3), "pitch": round(float(mad[1]), 3)},
        "tol": tol,
        "on_pfd_pct": round(100 * labels.count("on_pfd") / n, 1),
        "other_inside_pct": round(100 * labels.count("other_inside") / n, 1),
        "outside_pct": round(100 * labels.count("outside") / n, 1),
        "n_samples": len(labels),
        "samples": [{"t": g.t, "yaw": g.yaw, "pitch": g.pitch, "label": lb}
                    for g, lb in zip(s, labels)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--seat", default="left", help="which face-cam seat the cadet is (default left)")
    ap.add_argument("--tol", type=float, default=3.0, help="PFD membership width in MADs")
    ap.add_argument("--k-inside", type=int, default=3)
    args = ap.parse_args()

    track = GazeTrack.load(GAZE_DIR / args.sortie / f"{args.seat}.json")
    res = cluster(track, k_inside=args.k_inside, tol=args.tol)
    out = ATM_DIR / args.sortie / "gaze_pfd.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"seat={res['seat']}  PFD dir yaw={res['pfd_centroid']['yaw']}"
          f" pitch={res['pfd_centroid']['pitch']}  (MAD {res['pfd_mad']})")
    print(f"on PFD {res['on_pfd_pct']}%   other inside {res['other_inside_pct']}%"
          f"   outside {res['outside_pct']}%   ({res['n_samples']} samples)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
