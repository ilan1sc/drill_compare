"""11.1 AOI dwell time.

AOI = an unsupervised cluster of gaze directions (base.assign_aoi,
KMeans on yaw/pitch -- default 6 clusters), not a named instrument; see
schema.py's module docstring for why the face camera can't know which
gauge a cluster is. Once code/gaze/aoi_labels.py has recorded a
human-assigned name for a sortie+seat's clusters, this reports that name
too; until then, AOIs are just AOI_0.. AOI_k-1.

Per AOI: total dwell time, share of tracked time, mean dwell-per-visit,
and visit count -- and the same broken down using base.fixations()
fixation segments (dwell = time actually fixating inside the AOI, a
tighter number than raw sample-membership time).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import Result, fixations, run_cli  # noqa: E402


def compute(track) -> Result:
    fx = [f for f in fixations(track) if f["kind"] == "fixation" and f["aoi"] is not None]
    by_aoi: dict[int, list] = {}
    for f in fx:
        by_aoi.setdefault(f["aoi"], []).append(f)

    total_fix_time = sum(f["dur"] for f in fx) or 1e-9
    rows = []
    for aoi, fixs in sorted(by_aoi.items()):
        durs = np.array([f["dur"] for f in fixs])
        rows.append({
            "aoi": aoi, "aoi_label": fixs[0]["aoi_label"],
            "visits": len(fixs), "total_dwell_s": round(float(durs.sum()), 1),
            "share_of_fixation_time": round(float(durs.sum()) / total_fix_time, 3),
            "mean_dwell_per_visit_s": round(float(durs.mean()), 2),
        })
    rows.sort(key=lambda r: -r["total_dwell_s"])

    metrics = {"aois": len(rows), "table": rows}
    return Result("aoi_dwell_time", metrics, fx[:300],
                 limitations="AOI = unsupervised gaze cluster; see module docstring")


if __name__ == "__main__":
    run_cli("aoi_dwell_time", compute, needs_aoi=True)
