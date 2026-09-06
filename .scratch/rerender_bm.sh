#!/bin/bash
cd "/home/ilan/Documents/flight school/drill_compare"
DATA="/home/ilan/Documents/flight school/data"
for s in "s1 c2" "s1 c3" "s1 c4" "s1 c5" "s1 c6" "s1 c8"; do
  echo "======== $s $(date +%T) ========"
  python3 code/speech/audio_vis_biomarkers.py "$DATA/$s.mp4" \
      --dir "exports/audio_best/$s" --start 600 --duration 180 2>&1 \
      | grep -aE "\[sync\]|Wrote|Error|Traceback"
  rm -f "exports/audio_best/$s/"*.silent.mp4
done
echo "======== DONE $(date +%T) ========"
