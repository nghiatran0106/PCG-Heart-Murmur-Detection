#!/usr/bin/env bash
set -euo pipefail

cd ~/Nghia/PCG-Heart-Murmur-Detection
source .venv_pcg/bin/activate
export PYTHONPATH=$(pwd)
export PYTHONUNBUFFERED=1

PHASE="${1:-}"
LOG_DIR="outputs/experiment_logs/phase_by_phase"
STATUS_FILE="outputs/experiment_status/phase_by_phase_status.txt"

mkdir -p "$LOG_DIR"
mkdir -p outputs/experiment_status

usage() {
  echo "Usage:"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase0_prepare"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase1_bandpass"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase1_resampling"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase1_segmentation"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase1_spectrogram_enhancement"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase2_features"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase3_fusion"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase4_models"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase5_xai"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase6_multidataset"
  echo "  bash experiments/phase_plan/run_one_phase.sh phase7_optimization"
}

run_cmd() {
  local NAME="$1"
  local CMD="$2"
  local LOG_FILE="$LOG_DIR/${NAME}.log"

  echo "============================================================"
  echo "RUNNING PHASE: $NAME"
  echo "Command: $CMD"
  echo "Log: $LOG_FILE"
  echo "Start: $(date)"
  echo "============================================================"

  echo "[RUNNING] $NAME | $(date)" >> "$STATUS_FILE"

  set +e
  bash -lc "$CMD" 2>&1 | tee "$LOG_FILE"
  local RET=${PIPESTATUS[0]}
  set -e

  if [ "$RET" -eq 0 ]; then
    echo "[DONE] $NAME | $(date)" >> "$STATUS_FILE"
    echo "DONE: $NAME"
  else
    echo "[FAILED] $NAME | exit_code=$RET | $(date)" >> "$STATUS_FILE"
    echo "FAILED: $NAME"
    exit "$RET"
  fi
}

skip_missing() {
  local NAME="$1"
  local FILE="$2"

  echo "[SKIP] $NAME because missing: $FILE"
  echo "[SKIP] $NAME because missing: $FILE" >> "$STATUS_FILE"
}

if [ -z "$PHASE" ]; then
  usage
  exit 1
fi

case "$PHASE" in

  phase0_prepare)
    if [ -f scripts/build_unified_metadata.py ]; then
      run_cmd "phase0_build_unified_metadata" "python scripts/build_unified_metadata.py"
    else
      skip_missing "phase0_build_unified_metadata" "scripts/build_unified_metadata.py"
    fi

    if [ -f experiments/phase_plan/01_create_all_phase_configs.py ]; then
      run_cmd "phase0_create_all_phase_configs" "python experiments/phase_plan/01_create_all_phase_configs.py"
    else
      skip_missing "phase0_create_all_phase_configs" "experiments/phase_plan/01_create_all_phase_configs.py"
    fi
    ;;

  phase1_bandpass)
    if [ -f experiments/preprocessing_ablation/02_run_bandpass_ablation.sh ]; then
      run_cmd "phase1_bandpass" "bash experiments/preprocessing_ablation/02_run_bandpass_ablation.sh"
    else
      skip_missing "phase1_bandpass" "experiments/preprocessing_ablation/02_run_bandpass_ablation.sh"
    fi
    ;;

  phase1_resampling)
    if [ -f experiments/preprocessing_ablation/03_run_resampling_ablation.sh ]; then
      run_cmd "phase1_resampling" "bash experiments/preprocessing_ablation/03_run_resampling_ablation.sh"
    else
      skip_missing "phase1_resampling" "experiments/preprocessing_ablation/03_run_resampling_ablation.sh"
    fi
    ;;

  phase1_segmentation)
    if [ -f experiments/preprocessing_ablation/04_run_segmentation_ablation.sh ]; then
      run_cmd "phase1_segmentation" "bash experiments/preprocessing_ablation/04_run_segmentation_ablation.sh"
    else
      skip_missing "phase1_segmentation" "experiments/preprocessing_ablation/04_run_segmentation_ablation.sh"
    fi
    ;;

  phase1_spectrogram_enhancement)
    if [ -f experiments/preprocessing_ablation/05_run_spectrogram_enhancement_ablation.sh ]; then
      run_cmd "phase1_spectrogram_enhancement" "bash experiments/preprocessing_ablation/05_run_spectrogram_enhancement_ablation.sh"
    else
      skip_missing "phase1_spectrogram_enhancement" "experiments/preprocessing_ablation/05_run_spectrogram_enhancement_ablation.sh"
    fi
    ;;

  phase2_features)
    if [ -f experiments/feature_ablation/run_feature_ablation.sh ]; then
      run_cmd "phase2_features" "bash experiments/feature_ablation/run_feature_ablation.sh"
    else
      skip_missing "phase2_features" "experiments/feature_ablation/run_feature_ablation.sh"
    fi
    ;;

  phase3_fusion)
    if [ -f experiments/fusion_ablation/run_fusion_ablation.sh ]; then
      run_cmd "phase3_fusion" "bash experiments/fusion_ablation/run_fusion_ablation.sh"
    else
      skip_missing "phase3_fusion" "experiments/fusion_ablation/run_fusion_ablation.sh"
    fi
    ;;

  phase4_models)
    if [ -f experiments/model_block_ablation/run_model_block_ablation.sh ]; then
      run_cmd "phase4_models" "bash experiments/model_block_ablation/run_model_block_ablation.sh"
    else
      skip_missing "phase4_models" "experiments/model_block_ablation/run_model_block_ablation.sh"
    fi
    ;;

  phase5_xai)
    if [ -f experiments/xai_baseline/02_gradcam_spectrogram_branch.py ]; then
      run_cmd "phase5_gradcam" "for FOLD in 0 1 2 3 4; do python experiments/xai_baseline/02_gradcam_spectrogram_branch.py --fold \${FOLD} --max-per-class 4; done"
    else
      skip_missing "phase5_gradcam" "experiments/xai_baseline/02_gradcam_spectrogram_branch.py"
    fi

    if [ -f experiments/xai_baseline/02_occlusion_sensitivity.py ]; then
      run_cmd "phase5_occlusion" "for FOLD in 0 1 2 3 4; do python experiments/xai_baseline/02_occlusion_sensitivity.py --fold \${FOLD} --max-per-class 3; done"
    else
      skip_missing "phase5_occlusion" "experiments/xai_baseline/02_occlusion_sensitivity.py"
    fi

    if [ -f experiments/xai_baseline/03_tsne_pca_resnet_embedding.py ]; then
      run_cmd "phase5_tsne_pca" "for FOLD in 0 1 2 3 4; do python experiments/xai_baseline/03_tsne_pca_resnet_embedding.py --fold \${FOLD} --max-per-class 150; done"
    else
      skip_missing "phase5_tsne_pca" "experiments/xai_baseline/03_tsne_pca_resnet_embedding.py"
    fi

    if [ -f experiments/xai_baseline/05_handcrafted_weight_importance.py ]; then
      run_cmd "phase5_handcrafted_importance" "python experiments/xai_baseline/05_handcrafted_weight_importance.py"
    else
      skip_missing "phase5_handcrafted_importance" "experiments/xai_baseline/05_handcrafted_weight_importance.py"
    fi
    ;;

  phase6_multidataset)
    if [ -f experiments/multidataset_multitask/run_multidataset_multitask.sh ]; then
      run_cmd "phase6_multidataset" "bash experiments/multidataset_multitask/run_multidataset_multitask.sh"
    else
      skip_missing "phase6_multidataset" "experiments/multidataset_multitask/run_multidataset_multitask.sh"
    fi
    ;;

  phase7_optimization)
    if [ -f experiments/optimization/run_scheduler_loss_ablation.sh ]; then
      run_cmd "phase7_scheduler_loss" "bash experiments/optimization/run_scheduler_loss_ablation.sh"
    else
      skip_missing "phase7_scheduler_loss" "experiments/optimization/run_scheduler_loss_ablation.sh"
    fi

    if [ -f experiments/optuna/run_optuna_search.py ]; then
      run_cmd "phase7_optuna" "python experiments/optuna/run_optuna_search.py --config configs/experiments/phase7_optimization/optuna_search_fusion_baseline.yaml"
    else
      skip_missing "phase7_optuna" "experiments/optuna/run_optuna_search.py"
    fi
    ;;

  *)
    echo "Unknown phase: $PHASE"
    usage
    exit 1
    ;;
esac
