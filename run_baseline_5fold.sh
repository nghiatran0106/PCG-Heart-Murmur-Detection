#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk2/home/nhunglt/Nghia/sample/bts_murmur_detection
source .venv_pcg/bin/activate
export PYTHONPATH=$(pwd)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for FOLD in 0 1 2 3 4
do
  echo "============================================================"
  echo "START BASELINE FOLD ${FOLD}"
  echo "Time: $(date)"
  echo "============================================================"

  nice -n 15 ionice -c2 -n7 python src/training/train_baseline.py \
    --config configs/baseline_fusion.yaml \
    --fold ${FOLD} \
    --task outcome_binary \
    --epochs 40 \
    --batch-size 32 \
    --num-workers 2 \
    --amp

  echo "============================================================"
  echo "DONE BASELINE FOLD ${FOLD}"
  echo "Time: $(date)"
  echo "============================================================"
done
