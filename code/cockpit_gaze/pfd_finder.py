"""Find the PFD screen's pixel box in a 02_F2-PILOT CAM COCKPIT frame.

The rig moves between sessions, so the PFD isn't at a fixed pixel
location -- but it always looks the same: the RIGHTMOST of the two
bright Garmin displays in the centre console (map/nav display; the
left-hand round-dial six-pack screen is NOT the PFD -- confirmed
visually against a marked reference frame).

Method: threshold on brightness (the screens are the brightest thing in
frame besides the outside view, and we exclude the outside view by
restricting to the central-console ROI band), take the largest bright
rectangular blobs, and pick the leftmost one above a minimum size.

    python3 code/cockpit_gaze/pfd_finder.py "<02 video>" --out box.json --preview preview.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# fraction of frame where the console (not the windscreen/sky) lives --
# generous, so a moderate rig shift is still inside it
CONSOLE_ROI = (0.20, 0.35, 0.80, 0.75)   # x0,y0,x1,y1 fractions
MIN_AREA_FRAC = 0.01    # a real screen blob is at least this fraction of ROI area
BRIGHTNESS_THRESH = 140


def find_pfd(frame: np.ndarray, roi=CONSOLE_ROI, thresh=BRIGHTNESS_THRESH):
    h, w = frame.shape[:2]
    rx0, ry0, rx1, ry1 = (int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h))
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    roi_v = v[ry0:ry1, rx0:rx1]
    _, th = cv2.threshold(roi_v, thresh, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_FRAC * (rx1 - rx0) * (ry1 - ry0)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / max(bh, 1)
        if not (0.8 <= ar <= 2.2):   # a screen, not a thin glare streak
            continue
        boxes.append((x + rx0, y + ry0, bw, bh, area))
    if not boxes:
        return None
    boxes.sort(key=lambda b: b[4], reverse=True)
    big = boxes[:2] if len(boxes) >= 2 else boxes
    big.sort(key=lambda b: b[0])   # sort left->right; PFD is the rightmost of the two
    x, y, bw, bh, _ = big[-1]
    return (x, y, x + bw, y + bh)


def find_pfd_stable(video_path, n_samples=8):
    """Average the detection over several frames spread through the clip --
    more robust to a single frame's glare/hand-in-front-of-screen moment."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    dur = n / fps if fps else 0
    boxes = []
    for i in range(n_samples):
        t = dur * (i + 1) / (n_samples + 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        b = find_pfd(frame)
        if b:
            boxes.append(b)
    cap.release()
    if not boxes:
        return None
    arr = np.array(boxes)
    return tuple(int(v) for v in np.median(arr, axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", default=None, help="save an annotated frame with the box drawn in red")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    box = find_pfd_stable(args.video, args.samples)
    if box is None:
        raise SystemExit("no PFD-like bright rectangle found -- check the video / adjust CONSOLE_ROI")
    x0, y0, x1, y1 = box
    result = {"video": str(args.video), "box_xyxy": [x0, y0, x1, y1],
              "center": [(x0 + x1) / 2, (y0 + y1) / 2]}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    if args.preview:
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 6)
            cv2.imwrite(args.preview, frame)
            print("preview ->", args.preview)


if __name__ == "__main__":
    main()
