import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PRED_DIR = Path("outputs/predictions")
OUT_DIR = Path("outputs/report_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_fold_epoch(path: Path):
    m = re.search(r"fold(\d+)_epoch(\d+)_patients\.csv", path.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def normalize_target(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"0", "absent", "normal", "negative"}:
            return 0
        if v in {"1", "present", "abnormal", "positive"}:
            return 1

    return int(x)


def detect_columns(df: pd.DataFrame):
    target_candidates = [
        "target",
        "y_true",
        "true",
        "true_label",
        "label",
        "murmur_target",
        "outcome_target",
    ]

    pred_candidates = [
        "prediction",
        "pred",
        "y_pred",
        "pred_label",
    ]

    prob_candidates = [
        "prob_1",
        "prob_present",
        "present_prob",
        "probability",
        "score",
        "prob_positive",
        "prob_abnormal",
        "prob_Abnormal",
    ]

    target_col = next((c for c in target_candidates if c in df.columns), None)
    pred_col = next((c for c in pred_candidates if c in df.columns), None)
    prob_col = next((c for c in prob_candidates if c in df.columns), None)

    if target_col is None:
        raise KeyError(f"Cannot detect target column. Columns: {list(df.columns)}")

    if prob_col is None:
        raise KeyError(f"Cannot detect positive probability column. Columns: {list(df.columns)}")

    return target_col, pred_col, prob_col


def read_prediction_file(path: Path):
    df = pd.read_csv(path)
    target_col, pred_col, prob_col = detect_columns(df)

    y_true = df[target_col].apply(normalize_target).astype(int).values
    y_prob = df[prob_col].astype(float).values

    if pred_col is not None:
        y_pred = df[pred_col].apply(normalize_target).astype(int).values
    else:
        y_pred = (y_prob >= 0.5).astype(int)

    fold, epoch = parse_fold_epoch(path)

    return pd.DataFrame(
        {
            "source_file": str(path),
            "fold": fold,
            "epoch": epoch,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    )


def compute_metrics(pred_df: pd.DataFrame):
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    y_prob = pred_df["y_prob"].values

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    try:
        out["auroc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auroc"] = np.nan

    return out


def load_all_epoch_metrics():
    files = sorted(PRED_DIR.glob("fold*_epoch*_patients.csv"))
    if not files:
        raise FileNotFoundError(f"No patient prediction files found in {PRED_DIR}")

    metric_rows = []

    for path in files:
        fold, epoch = parse_fold_epoch(path)
        if fold is None or epoch is None:
            continue

        try:
            pred_df = read_prediction_file(path)
            metrics = compute_metrics(pred_df)
        except Exception as e:
            print(f"[SKIP] {path}: {e}")
            continue

        metric_rows.append(
            {
                "file": str(path),
                "fold": fold,
                "epoch": epoch,
                **metrics,
            }
        )

    if not metric_rows:
        raise RuntimeError("No valid prediction files were parsed.")

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT_DIR / "all_epoch_patient_metrics.csv", index=False)
    return metric_df


def select_best_epoch_per_fold(metric_df: pd.DataFrame):
    best_rows = []

    for fold, sub in metric_df.groupby("fold"):
        sub = sub.copy()
        sub = sub.sort_values(
            by=["f1", "auroc", "accuracy"],
            ascending=[False, False, False],
        )
        best_rows.append(sub.iloc[0])

    best_df = pd.DataFrame(best_rows).sort_values("fold")
    best_df.to_csv(OUT_DIR / "selected_best_epoch_per_fold.csv", index=False)

    pred_parts = []
    for _, row in best_df.iterrows():
        pred_parts.append(read_prediction_file(Path(row["file"])))

    combined = pd.concat(pred_parts, ignore_index=True)
    combined.to_csv(OUT_DIR / "combined_best_fold_predictions.csv", index=False)

    return best_df, combined


def plot_confusion_matrix(pred_df: pd.DataFrame):
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Absent", "Present"])
    ax.set_yticklabels(["Absent", "Present"])

    for i in range(2):
        for j in range(2):
            label = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"
            text_color = "white" if cm_norm[i, j] < 0.35 else "black"
            ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion")

    fig.tight_layout()
    out = OUT_DIR / "figure_confusion_matrix.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved:", out)


def plot_roc_curve(best_df: pd.DataFrame, combined_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))

    for _, row in best_df.iterrows():
        pred_df = read_prediction_file(Path(row["file"]))
        y_true = pred_df["y_true"].values
        y_prob = pred_df["y_prob"].values

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = roc_auc_score(y_true, y_prob)

        ax.plot(fpr, tpr, linewidth=1.2, alpha=0.65, label=f"Fold {int(row['fold'])} AUC={fold_auc:.3f}")

    y_true = combined_df["y_true"].values
    y_prob = combined_df["y_prob"].values
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pooled_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, linewidth=2.8, label=f"Pooled AUC={pooled_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random classifier")

    ax.set_title("ROC Curve Across Five Folds")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "figure_roc_curve.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved:", out)


def plot_learning_curve_by_fold(metric_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    for fold, sub in metric_df.groupby("fold"):
        sub = sub.sort_values("epoch")
        ax.plot(
            sub["epoch"].values,
            sub["f1"].values,
            marker="o",
            linewidth=1.3,
            alpha=0.7,
            label=f"Fold {int(fold)}",
        )

    summary = (
        metric_df.groupby("epoch")
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
        .sort_values("epoch")
        .fillna(0.0)
    )

    x = summary["epoch"].values
    mean = summary["f1_mean"].values
    std = summary["f1_std"].values

    ax.plot(x, mean, marker="s", linewidth=3.0, label="Mean F1")
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, label="Mean ± std")

    ax.set_title("Validation Learning Curve by Fold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Patient-level F1-score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "figure_learning_curve.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved:", out)


def main():
    metric_df = load_all_epoch_metrics()
    best_df, combined_df = select_best_epoch_per_fold(metric_df)

    print("\n===== Selected best epoch per fold =====")
    print(
        best_df[
            ["fold", "epoch", "accuracy", "precision", "recall", "f1", "auroc"]
        ].round(4).to_string(index=False)
    )

    print("\n===== Combined best-fold metrics =====")
    combined_metrics = compute_metrics(combined_df)
    for k, v in combined_metrics.items():
        print(f"{k}: {v:.4f}")

    plot_confusion_matrix(combined_df)
    plot_roc_curve(best_df, combined_df)
    plot_learning_curve_by_fold(metric_df)


if __name__ == "__main__":
    main()
