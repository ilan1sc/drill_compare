#!/bin/bash
cd "/home/ilan/Documents/flight school/drill_compare"
DATA="/home/ilan/Documents/flight school/data"
WIN_START=600
WIN_DUR=180
for s in "s1 c2" "s1 c3" "s1 c4" "s1 c5" "s1 c6" "s1 c8"; do
  echo "======== $s $(date +%T) ========"
  if [ ! -f "exports/audio_best/$s/biomarkers/_summary.json" ]; then
    python3 code/speech/pipeline_best.py "$DATA/$s.mp4" 2>&1 | grep -avE "Loading weights|it/s\]$|^\[transformers\]"
  else
    echo "  pipeline already done, skipping"
  fi
  python3 code/speech/audio_vis_biomarkers.py "$DATA/$s.mp4" \
      --dir "exports/audio_best/$s" --start $WIN_START --duration $WIN_DUR 2>&1 \
      | grep -aoE "Wrote .*|Error.*|Traceback.*"
  rm -f "exports/audio_best/$s/"*.silent.mp4
done
echo "======== ALL DONE $(date +%T) ========"
