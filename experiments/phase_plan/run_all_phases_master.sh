#!/usr/bin/env bash
set -e

cd ~/Nghia/PCG-Heart-Murmur-Detection
source .venv_pcg/bin/activate
export PYTHONPATH=$(pwd)
export PYTHONUNBUFFERED=1

GPU_ID=${1:-0}
MEM_LIMIT_MB=${2:-2000}

MASTER_LOG="outputs/experiment_logs/master/all_phases_master.log"
STATUS_FILE="outputs/experiment_status/all_phases_status.txt"
TODO_FILE="outputs/experiment_status/skipped_unimplemented_jobs.txt"

mkdir -p outputs/experiment_logs/master
mkdir -p outputs/experiment_status

echo "============================================================" | tee "$MASTER_LOG"
echo "PCG ALL PHASES MASTER RUN" | tee -a "$MASTER_LOG"
echo "Start time: $(date)" | tee -a "$MASTER_LOG"
echo "GPU_ID=$GPU_ID | MEM_LIMIT_MB=$MEM_LIMIT_MB" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

echo "" > "$STATUS_FILE"
echo "" > "$TODO_FILE"

wait_gpu() {
  echo "Waiting for GPU $GPU_ID memory < ${MEM_LIMIT_MB} MiB ..." | tee -a "$MASTER_LOG"

  while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
    echo "$(date) | GPU $GPU_ID used memory: ${USED} MiB" | tee -a "$MASTER_LOG"

    if [ "$USED" -lt "$MEM_LIMIT_MB" ]; then
      echo "GPU is free enough. Continue." | tee -a "$MASTER_LOG"
      break
    fi

    sleep 300
  done
}

run_cmd() {
  NAME="$1"
  CMD="$2"
  LOG_FILE="outputs/experiment_logs/master/${NAME}.log"

  echo "" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"
  echo "RUNNING: $NAME" | tee -a "$MASTER_LOG"
  echo "Command: $CMD" | tee -a "$MASTER_LOG"
  echo "Log: $LOG_FILE" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"

  echo "[RUNNING] $NAME | $(date)" >> "$STATUS_FILE"

  set +e
  bash -lc "$CMD" 2>&1 | tee "$LOG_FILE"
  RET=${PIPESTATUS[0]}
  set -e

  if [ "$RET" -eq 0 ]; then
    echo "[DONE] $NAME | $(date)" | tee -a "$MASTER_LOG"
    echo "[DONE] $NAME | $(date)" >> "$STATUS_FILE"
  else
    echo "[FAILED] $NAME | exit_code=$RET | $(date)" | tee -a "$MASTER_LOG"
    echo "[FAILED] $NAME | exit_code=$RET | $(date)" >> "$STATUS_FILE"
    exit "$RET"
  fi
}

skip_job() {
  NAME="$1"
  REASON="$2"
  echo "[SKIP] $NAME -- $REASON" | tee -a "$MASTER_LOG"
  echo "[SKIP] $NAME -- $REASON" >> "$STATUS_FILE"
  echo "$NAME: $REASON" >> "$TODO_FILE"
}

wait_gpu

# ============================================================
# Phase 0: metadata + config generation
# ============================================================

if [ -f scripts/build_unified_metadata.py ]; then
  run_cmd "phase0_build_unified_metadata" "python scripts/build_unified_metadata.py"
else
  skip_job "phase0_build_unified_metadata" "scripts/build_unified_metadata.py not found"
fi

if [ -f experiments/phase_plan/01_create_all_phase_configs.py ]; then
  run_cmd "phase0_create_all_phase_configs" "python experiments/phase_plan/01_create_all_phase_configs.py"
else
  skip_job "phase0_create_all_phase_configs" "experiments/phase_plan/01_create_all_phase_configs.py not found"
fi

# ============================================================
# Phase 1: preprocessing ablation
# ============================================================

# IMPORTANT:
# Only run configs that are supported by current train_baseline.py.
# If preprocessing regeneration script exists, run it first.
# Otherwise, run training configs only if they point to already-built processed CSVs.

if [ -f experiments/preprocessing_ablation/build_preprocessing_ablation_segments.py ]; then
  run_cmd "phase1_build_preprocessing_ablation_segments" "python experiments/preprocessing_ablation/build_preprocessing_ablation_segments.py"
else
  skip_job "phase1_build_preprocessing_ablation_segments" "not implemented yet; needed for fully valid bandpass/resampling/segmentation ablation"
fi

if [ -f experiments/preprocessing_ablation/02_run_bandpass_ablation.sh ]; then
  run_cmd "phase1_bandpass_ablation" "bash experiments/preprocessing_ablation/02_run_bandpass_ablation.sh"
else
  skip_job "phase1_bandpass_ablation" "runner not found"
fi

if [ -f experiments/preprocessing_ablation/03_run_resampling_ablation.sh ]; then
  run_cmd "phase1_resampling_ablation" "bash experiments/preprocessing_ablation/03_run_resampling_ablation.sh"
else
  skip_job "phase1_resampling_ablation" "runner not implemented yet"
fi

if [ -f experiments/preprocessing_ablation/04_run_segmentation_ablation.sh ]; then
  run_cmd "phase1_segmentation_ablation" "bash experiments/preprocessing_ablation/04_run_segmentation_ablation.sh"
else
  skip_job "phase1_segmentation_ablation" "runner not implemented yet"
fi

if [ -f experiments/preprocessing_ablation/05_run_spectrogram_enhancement_ablation.sh ]; then
  run_cmd "phase1_spectrogram_enhancement_ablation" "bash experiments/preprocessing_ablation/05_run_spectrogram_enhancement_ablation.sh"
else
  skip_job "phase1_spectrogram_enhancement_ablation" "runner not implemented yet"
fi

# ============================================================
# Phase 2: feature ablation
# ============================================================

if [ -f experiments/feature_ablation/run_feature_ablation.sh ]; then
  run_cmd "phase2_feature_ablation" "bash experiments/feature_ablation/run_feature_ablation.sh"
else
  skip_job "phase2_feature_ablation" "feature ablation runner not implemented yet"
fi

# ============================================================
# Phase 3: fusion ablation
# ============================================================

if [ -f experiments/fusion_ablation/run_fusion_ablation.sh ]; then
  run_cmd "phase3_fusion_ablation" "bash experiments/fusion_ablation/run_fusion_ablation.sh"
else
  skip_job "phase3_fusion_ablation" "fusion ablation runner not implemented yet"
fi

# ============================================================
# Phase 4: model block ablation
# ============================================================

if [ -f experiments/model_block_ablation/run_model_block_ablation.sh ]; then
  run_cmd "phase4_model_block_ablation" "bash experiments/model_block_ablation/run_model_block_ablation.sh"
else
  skip_job "phase4_model_block_ablation" "model block ablation runner not implemented yet"
fi

# ============================================================
# Phase 5: XAI
# ============================================================

if [ -f experiments/xai_baseline/02_gradcam_spectrogram_branch.py ]; then
  run_cmd "phase5_gradcam" "for FOLD in 0 1 2 3 4; do python experiments/xai_baseline/02_gradcam_spectrogram_branch.py --fold \$FOLD --max-per-class 4; done"
else
  skip_job "phase5_gradcam" "Grad-CAM script not found"
fi

if [ -f experiments/xai_baseline/02_occlusion_sensitivity.py ]; then
  run_cmd "phase5_occlusion_sensitivity" "for FOLD in 0 1 2 3 4; do python experiments/xai_baseline/02_occlusion_sensitivity.py --fold \$FOLD --max-per-class 3; done"
else
  skip_job "phase5_occlusion_sensitivity" "occlusion sensitivity script not found"
fi

if [ -f experiments/xai_baseline/03_tsne_pca_resnet_embedding.py ]; then
  run_cmd "phase5_tsne_pca" "for FOLD in 0 1 2 3 4; do python experiments/xai_baseline/03_tsne_pca_resnet_embedding.py --fold \$FOLD --max-per-class 150; done"
else
  skip_job "phase5_tsne_pca" "t-SNE/PCA script not found"
fi

if [ -f experiments/xai_baseline/05_handcrafted_weight_importance.py ]; then
  run_cmd "phase5_handcrafted_importance" "python experiments/xai_baseline/05_handcrafted_weight_importance.py"
else
  skip_job "phase5_handcrafted_importance" "handcrafted importance script not found"
fi

# ============================================================
# Phase 6: multi-dataset + multitask
# ============================================================

if [ -f experiments/multidataset_multitask/run_multidataset_multitask.sh ]; then
  run_cmd "phase6_multidataset_multitask" "bash experiments/multidataset_multitask/run_multidataset_multitask.sh"
else
  skip_job "phase6_multidataset_multitask" "multi-dataset/multitask runner not implemented yet"
fi

# ============================================================
# Phase 7: optimization / Optuna
# ============================================================

if [ -f experiments/optimization/run_scheduler_loss_ablation.sh ]; then
  run_cmd "phase7_scheduler_loss_optimization" "bash experiments/optimization/run_scheduler_loss_ablation.sh"
else
  skip_job "phase7_scheduler_loss_optimization" "scheduler/loss ablation runner not implemented yet"
fi

if [ -f experiments/optuna/run_optuna_search.py ]; then
  run_cmd "phase7_optuna" "python experiments/optuna/run_optuna_search.py --config configs/experiments/phase7_optimization/optuna_search_fusion_baseline.yaml"
else
  skip_job "phase7_optuna" "Optuna runner not implemented yet"
fi

echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "ALL RUNNABLE PHASES FINISHED" | tee -a "$MASTER_LOG"
echo "End time: $(date)" | tee -a "$MASTER_LOG"
echo "Status file: $STATUS_FILE" | tee -a "$MASTER_LOG"
echo "Skipped/TODO file: $TODO_FILE" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
