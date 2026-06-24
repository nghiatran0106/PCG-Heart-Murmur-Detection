#!/usr/bin/env bash
set -e

cd ~/Nghia/PCG-Heart-Murmur-Detection
source .venv_pcg/bin/activate
export PYTHONPATH=$(pwd)

CONFIGS=(
  "configs/ablation_preprocessing/bp_none_sr4000_seg5.yaml"
  "configs/ablation_preprocessing/bp_20_400_sr4000_seg5.yaml"
  "configs/ablation_preprocessing/bp_25_400_sr4000_seg5.yaml"
  "configs/ablation_preprocessing/bp_25_800_sr4000_seg5.yaml"
  "configs/ablation_preprocessing/bp_50_800_sr4000_seg5.yaml"
)

for CFG in "${CONFIGS[@]}"; do
  EXP_NAME=$(basename "$CFG" .yaml)

  echo "============================================================"
  echo "Running preprocessing ablation: $EXP_NAME"
  echo "Config: $CFG"
  echo "============================================================"

  for FOLD in 0 1 2 3 4; do
    echo "---------- $EXP_NAME | fold $FOLD ----------"

    python src/training/train_baseline.py \
      --config "$CFG" \
      --fold "$FOLD"
  done
done
