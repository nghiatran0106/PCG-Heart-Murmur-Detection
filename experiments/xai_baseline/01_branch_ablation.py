import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.training.train_baseline import YAMLConfigLoader, ReproducibilityManager, DataModule
from src.models.resnet_fusion import FusionResNetClassifier


def find_checkpoint(output_dir: str, fold: int) -> Path:
    candidates = [
        Path(output_dir) / "checkpoints" / f"best_fold{fold}.pt",
        Path(output_dir) / "checkpoints" / f"baseline_fusion_fold{fold}_best.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def load_checkpoint(path: Path, device):
    return torch.load(path, map_location=device, weights_only=False)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
        else:
            state = checkpoint
    else:
        state = checkpoint

    fixed = {}
    for k, v in state.items():
        if k.startswith("module."):
            fixed[k[len("module."):]] = v
        else:
            fixed[k] = v

    return fixed


def detect_handcrafted_dim(state_dict):
    key = "hand_branch.0.weight"
    if key not in state_dict:
        candidates = [k for k in state_dict.keys() if "hand_branch" in k and k.endswith("weight")]
        raise KeyError(
            f"Cannot detect handcrafted_dim from checkpoint. Missing {key}. "
            f"Candidates: {candidates[:20]}"
        )

    return int(state_dict[key].shape[1])


def align_handcrafted(handcrafted, expected_dim):
    current_dim = handcrafted.shape[1]

    if current_dim == expected_dim:
        return handcrafted, None

    if current_dim < expected_dim:
        pad = expected_dim - current_dim
        handcrafted = F.pad(handcrafted, (0, pad), mode="constant", value=0.0)
        return handcrafted, f"handcrafted feature padded from {current_dim} to {expected_dim}"

    handcrafted = handcrafted[:, :expected_dim]
    return handcrafted, f"handcrafted feature cropped from {current_dim} to {expected_dim}"


def binary_metrics(y_true, prob_pos):
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos).astype(float)
    y_pred = (prob_pos >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp + 1e-8)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    try:
        metrics["auroc"] = roc_auc_score(y_true, prob_pos)
    except Exception:
        metrics["auroc"] = float("nan")

    try:
        metrics["average_precision"] = average_precision_score(y_true, prob_pos)
    except Exception:
        metrics["average_precision"] = float("nan")

    return metrics


@torch.no_grad()
def evaluate_mode(model, loader, device, mode, expected_hand_dim):
    model.eval()

    rows = []
    warning_message = None

    for batch_idx, batch in enumerate(loader):
        logmel = batch["logmel"].to(device, non_blocking=True)
        handcrafted = batch["handcrafted"].to(device, non_blocking=True)
        target = batch["target"].detach().cpu().numpy().astype(int)

        handcrafted, msg = align_handcrafted(handcrafted, expected_hand_dim)
        if msg is not None and warning_message is None:
            warning_message = msg

        if mode == "zero_logmel":
            logmel = torch.zeros_like(logmel)

        if mode == "zero_handcrafted":
            handcrafted = torch.zeros_like(handcrafted)

        logits = model(logmel, handcrafted)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()

        patient_ids = batch.get("patient_id", None)
        if patient_ids is None:
            patient_ids = [f"sample_{batch_idx}_{i}" for i in range(len(target))]

        for i in range(len(target)):
            rows.append(
                {
                    "patient_id": str(patient_ids[i]),
                    "target": int(target[i]),
                    "prob_0": float(probs[i, 0]),
                    "prob_1": float(probs[i, 1]),
                }
            )

    seg_df = pd.DataFrame(rows)

    patient_df = (
        seg_df.groupby("patient_id", sort=False)
        .agg(
            target=("target", "first"),
            prob_0=("prob_0", "mean"),
            prob_1=("prob_1", "mean"),
        )
        .reset_index()
    )

    metrics = binary_metrics(patient_df["target"].values, patient_df["prob_1"].values)
    return metrics, patient_df, warning_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_fusion.yaml")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--task", default="murmur_binary")
    args = parser.parse_args()

    cfg = YAMLConfigLoader.load(args.config, args.fold)
    cfg.task = args.task
    cfg.num_classes = 2

    ReproducibilityManager.set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = find_checkpoint(cfg.output_dir, cfg.fold)
    checkpoint = load_checkpoint(ckpt_path, device)
    state_dict = extract_state_dict(checkpoint)

    checkpoint_hand_dim = detect_handcrafted_dim(state_dict)

    print("=" * 80)
    print(f"Fold: {cfg.fold}")
    print(f"Task: {cfg.task}")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Config handcrafted_dim: {cfg.handcrafted_dim}")
    print(f"Checkpoint handcrafted_dim: {checkpoint_hand_dim}")
    print("=" * 80)

    data_module = DataModule(cfg)
    _, val_loader, _, _ = data_module.setup()

    model = FusionResNetClassifier(
        handcrafted_dim=checkpoint_hand_dim,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)

    out_dir = Path("outputs/xai_baseline/branch_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["full", "zero_logmel", "zero_handcrafted"]
    result_rows = []
    warnings = []

    for mode in modes:
        metrics, pred_df, warning_message = evaluate_mode(
            model=model,
            loader=val_loader,
            device=device,
            mode=mode,
            expected_hand_dim=checkpoint_hand_dim,
        )

        if warning_message is not None:
            warnings.append(warning_message)

        metrics["fold"] = cfg.fold
        metrics["mode"] = mode
        result_rows.append(metrics)

        pred_path = out_dir / f"fold{cfg.fold}_{mode}_patient_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

    result = pd.DataFrame(result_rows)
    result = result[
        [
            "fold",
            "mode",
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1",
            "balanced_accuracy",
            "mcc",
            "kappa",
            "auroc",
            "average_precision",
        ]
    ]

    out_path = out_dir / f"fold{cfg.fold}_branch_ablation.csv"
    result.to_csv(out_path, index=False)

    if warnings:
        print("\nWARNING:")
        for w in sorted(set(warnings)):
            print(f"- {w}")
        print(
            "\nThis means the checkpoint expects a different handcrafted feature dimension "
            "than the current DataModule produces. The script aligned the tensor to avoid crashing."
        )

    print("\n===== BRANCH ABLATION RESULT =====")
    print(result.round(4).to_string(index=False))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
