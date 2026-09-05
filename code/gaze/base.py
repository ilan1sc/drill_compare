"""Shared plumbing for the video/gaze biomarkers (PRD 11.1).

Every module here consumes a schema.GazeTrack (one seat, one sortie --
see extract_gaze.py) and exposes `compute(track, **kwargs) -> Result`
plus a CLI: `python3 code/gaze/fixation_duration_frequency.py
exports/gaze/<sortie>/<seat>.json`.

Fixation/saccade detection uses I-VT (velocity threshold): angular
velocity between consecutive samples, thresholded at this TRACK's own
median + k*MAD (scenario-relative, the same convention as
b14_pause_hesitation.py and controls_biomarkers/base.py) rather than the
fixed ~30-100 deg/s literature cutoffs, which assume a real eye-tracker's
sample rate and noise floor -- ours is a 5Hz appearance-based proxy, not
a calibrated eye-tracker, so an absolute threshold would be meaningless.

AOI = an unsupervised cluster of gaze directions (KMeans on yaw/pitch),
NOT a named instrument -- the camera watches the crew, not what the crew
sees, so no automatic pipeline can know which cluster is which gauge.
`aoi_labels.py` is the (manual, one-time-per-seat) calibration step that
attaches real names; every AOI-consuming module works before that step
too, just with generic "AOI_0" style ids.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import GazeTrack  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
GAZE_DIR = ROOT / "exports" / "gaze"
MAX_GAP_S = 1.0          # bigger gaps = tracking lost, not a real saccade -- excluded
MIN_FIXATION_S = 0.1     # fixations shorter than this are noise, not real dwell


@dataclass
class Result:
    name: str
    metrics: dict
    evidence: list[dict] = field(default_factory=list)
    limitations: str = ""

    def save(self, sortie: str, seat: str, root: Path = GAZE_DIR) -> Path:
        out = root / sortie / seat / f"{self.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
        return out


def robust_threshold(x: np.ndarray, k: float = 2.5) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-9
    return float(med + k * mad)


def valid_pairs(track: GazeTrack):
    """consecutive samples with a real (not tracking-loss) gap between them."""
    s = track.samples
    for a, b in zip(s, s[1:]):
        if 0 < (b.t - a.t) <= MAX_GAP_S:
            yield a, b


def angular_velocity(track: GazeTrack) -> list[tuple[float, float, float]]:
    """[(t_mid, dt, deg/s)] between consecutive valid samples."""
    out = []
    for a, b in valid_pairs(track):
        dt = b.t - a.t
        dist = float(np.hypot(b.yaw - a.yaw, b.pitch - a.pitch))
        out.append(((a.t + b.t) / 2, dt, dist / dt))
    return out


def fixations(track: GazeTrack, k: float = 2.5):
    """I-VT fixation/saccade segmentation. Returns list of dicts:
    {kind: 'fixation'|'saccade', t0, t1, dur, yaw, pitch, samples}."""
    vel = angular_velocity(track)
    if not vel:
        return []
    thresh = robust_threshold(np.array([v[2] for v in vel]), k)

    s = track.samples
    segs, cur_kind, cur = [], None, []
    idx_by_t = {round(a.t, 3): i for i, a in enumerate(s)}
    for i in range(1, len(s)):
        a, b = s[i - 1], s[i]
        gap = b.t - a.t
        if not (0 < gap <= MAX_GAP_S):
            if cur:
                segs.append((cur_kind, cur))
            cur, cur_kind = [], None
            continue
        v = float(np.hypot(b.yaw - a.yaw, b.pitch - a.pitch)) / gap
        kind = "fixation" if v <= thresh else "saccade"
        if kind != cur_kind and cur:
            segs.append((cur_kind, cur))
            cur = []
        cur_kind = kind
        if not cur:
            cur.append(a)
        cur.append(b)
    if cur:
        segs.append((cur_kind, cur))

    out = []
    for kind, pts in segs:
        dur = pts[-1].t - pts[0].t
        if kind == "fixation" and dur < MIN_FIXATION_S:
            continue
        out.append({
            "kind": kind, "t0": round(pts[0].t, 2), "t1": round(pts[-1].t, 2),
            "dur": round(dur, 2),
            "yaw": round(float(np.mean([p.yaw for p in pts])), 2),
            "pitch": round(float(np.mean([p.pitch for p in pts])), 2),
            "samples": len(pts), "aoi": pts[len(pts) // 2].aoi,
            "aoi_label": pts[len(pts) // 2].aoi_label,
        })
    return out


def assign_aoi(track: GazeTrack, k: int = 6) -> dict:
    """KMeans on (yaw, pitch) -> track.samples[i].aoi in place. Returns
    {centroids: [[yaw,pitch],...], sizes: [...]}. Labels (aoi_label) are
    filled from a calibration file if aoi_labels.py has been run for this
    sortie/seat, else left None (generic AOI_<n> ids only)."""
    from sklearn.cluster import KMeans

    pts = np.array([[s.yaw, s.pitch] for s in track.samples])
    if len(pts) < k:
        k = max(1, len(pts) // 3) or 1
    if k < 1 or len(pts) == 0:
        return {"centroids": [], "sizes": []}
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(pts)
    for s, lab in zip(track.samples, km.labels_):
        s.aoi = int(lab)

    labels = load_aoi_labels(track.sortie, track.seat)
    if labels:
        for s in track.samples:
            s.aoi_label = labels.get(str(s.aoi))

    sizes = [int((km.labels_ == i).sum()) for i in range(k)]
    return {"centroids": km.cluster_centers_.round(2).tolist(), "sizes": sizes}


def search_episodes(fx: list[dict], target_aoi: int) -> list[dict]:
    """Every return-to-`target_aoi` in a fixation sequence (base.fixations,
    'fixation' kind only), as a search episode: from departing the AOI
    visited just before, to arriving back at the target. Returns
    [{target, start, end, duration, aois_visited: [...], n_aois_visited}]
    -- see search_time.py / incorrect_search_locations.py."""
    episodes = []
    i, n = 0, len(fx)
    while i < n:
        if fx[i]["aoi"] != target_aoi:
            i += 1
            continue
        j, visited = i - 1, []
        while j >= 0 and fx[j]["aoi"] != target_aoi:
            visited.append(fx[j]["aoi"])
            j -= 1
        if j >= 0:
            start_t = fx[j]["t1"]
        elif visited:
            start_t = fx[0]["t0"]
        else:
            i += 1
            continue
        end_t = fx[i]["t0"]
        episodes.append({
            "target": target_aoi, "start": round(start_t, 2), "end": round(end_t, 2),
            "duration": round(end_t - start_t, 2), "aois_visited": list(reversed(visited)),
            "n_aois_visited": len(set(visited)),
        })
        while i < n and fx[i]["aoi"] == target_aoi:
            i += 1
    return episodes


def load_aoi_labels(sortie: str, seat: str) -> dict | None:
    p = GAZE_DIR / sortie / seat / "aoi_labels.json"
    return json.loads(p.read_text()) if p.exists() else None


def run_cli(module_name: str, compute_fn, needs_aoi: bool = False):
    import argparse
    ap = argparse.ArgumentParser(description=f"Gaze biomarker: {module_name}")
    ap.add_argument("gaze_json", help="exports/gaze/<sortie>/<seat>.json")
    ap.add_argument("--aoi-k", type=int, default=6)
    args = ap.parse_args()

    track = GazeTrack.load(args.gaze_json)
    if needs_aoi and (not track.samples or track.samples[0].aoi is None):
        assign_aoi(track, k=args.aoi_k)
    result = compute_fn(track)
    out = result.save(track.sortie, track.seat)
    print(f"{module_name} [{track.sortie}/{track.seat}]: "
          f"{json.dumps(result.metrics, ensure_ascii=False)[:500]}")
    print(f"Wrote {out}")
