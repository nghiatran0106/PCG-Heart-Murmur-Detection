#!/usr/bin/env bash
set -e

cd ~/Nghia/PCG-Heart-Murmur-Detection
source .venv_pcg/bin/activate
export PYTHONPATH=$(pwd)

GPU_ID=${1:-0}
MEM_LIMIT_MB=${2:-2000}

echo "Watching GPU $GPU_ID until used memory < ${MEM_LIMIT_MB} MiB"

while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
  echo "$(date) | GPU $GPU_ID used memory: ${USED} MiB"

  if [ "$USED" -lt "$MEM_LIMIT_MB" ]; then
    echo "GPU is free enough. Starting supported baseline jobs."
    bash outputs/experiment_queue/run_supported_baseline_jobs_later.sh
    break
  fi

  sleep 300
done
