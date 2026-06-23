from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


OUT_DIR = Path("outputs/xai_baseline/handcrafted_weight_importance")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_state_dict(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    fixed = {}
    for k, v in state.items():
        fixed[k.replace("module.", "")] = v.detach().cpu()
    return fixed


def detect_dims(state):
    hand_input_dim = int(state["hand_branch.0.weight"].shape[1])
    hand_hidden_dim = int(state["hand_branch.0.weight"].shape[0])
    hand_embed_dim = int(state["hand_branch.4.weight"].shape[0])
    deep_dim = int(state["deep_branch.backbone.fc.weight"].shape[0])
    fusion_dim = int(state["classifier.0.weight"].shape[0])
    classifier_hidden_dim = int(state["classifier.2.weight"].shape[0])
    num_classes = int(state["classifier.5.weight"].shape[0])

    return {
        "hand_input_dim": hand_input_dim,
        "hand_hidden_dim": hand_hidden_dim,
        "hand_embed_dim": hand_embed_dim,
        "deep_dim": deep_dim,
        "fusion_dim": fusion_dim,
        "classifier_hidden_dim": classifier_hidden_dim,
        "num_classes": num_classes,
    }


def compute_importance(state):
    W1 = state["hand_branch.0.weight"].numpy()      # [128, 144]
    W2 = state["hand_branch.4.weight"].numpy()      # [128, 128]
    Wc = state["classifier.2.weight"].numpy()       # [256, 384]
    Wo = state["classifier.5.weight"].numpy()       # [2, 256]

    hand_input_dim = W1.shape[1]
    deep_dim = state["deep_branch.backbone.fc.weight"].shape[0]
    hand_embed_dim = W2.shape[0]

    # 1. Direct input sensitivity: how strongly each handcrafted input
    # connects to the first MLP hidden layer.
    first_layer_l2 = np.linalg.norm(W1, ord=2, axis=0)
    first_layer_l1 = np.sum(np.abs(W1), axis=0)

    # 2. Approximate class-contrast path importance:
    # handcrafted input -> hand MLP -> handcrafted embedding -> fusion classifier -> Present vs Absent.
    # This ignores non-linearities, BatchNorm, Dropout, and LayerNorm, so it is an approximate
    # parameter-based importance score, not a causal or permutation-based importance score.
    Wc_hand = Wc[:, deep_dim:deep_dim + hand_embed_dim]  # [256, 128]
    class_contrast = Wo[1] - Wo[0]                       # [256]

    effect_hand_embed = class_contrast @ Wc_hand         # [128]
    effect_hidden = effect_hand_embed @ W2               # [128]
    effect_input = effect_hidden @ W1                    # [144]

    path_importance_abs = np.abs(effect_input)
    path_importance_signed = effect_input

    df = pd.DataFrame({
        "feature_index": np.arange(hand_input_dim),
        "feature_name": [f"handcrafted_feature_{i:03d}" for i in range(hand_input_dim)],
        "first_layer_l2": first_layer_l2,
        "first_layer_l1": first_layer_l1,
        "path_importance_abs": path_importance_abs,
        "path_importance_signed": path_importance_signed,
    })

    for col in ["first_layer_l2", "first_layer_l1", "path_importance_abs"]:
        s = df[col].values
        df[col + "_norm"] = s / (s.max() + 1e-8)

    return df


def plot_top_features(df, score_col, title, out_path, top_k=30):
    top = df.sort_values(score_col, ascending=False).head(top_k).copy()
    top = top.sort_values(score_col, ascending=True)

    labels = top["feature_name"].values
    values = top[score_col].values

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel(score_col)
    ax.set_ylabel("Handcrafted feature index")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("Saved:", out_path)


def summarize_across_folds():
    all_rows = []

    for fold in range(5):
        ckpt_path = Path(f"outputs/checkpoints/best_fold{fold}.pt")
        if not ckpt_path.exists():
            print(f"Skip missing checkpoint: {ckpt_path}")
            continue

        state = load_state_dict(ckpt_path)
        dims = detect_dims(state)
        df = compute_importance(state)
        df["fold"] = fold

        for k, v in dims.items():
            df[k] = v

        fold_path = OUT_DIR / f"fold{fold}_handcrafted_weight_importance.csv"
        df.to_csv(fold_path, index=False)
        print("Saved:", fold_path)

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No checkpoints found.")

    all_df = pd.concat(all_rows, ignore_index=True)

    summary = (
        all_df.groupby(["feature_index", "feature_name"])
        .agg(
            first_layer_l2_mean=("first_layer_l2_norm", "mean"),
            first_layer_l2_std=("first_layer_l2_norm", "std"),
            path_importance_mean=("path_importance_abs_norm", "mean"),
            path_importance_std=("path_importance_abs_norm", "std"),
            signed_path_mean=("path_importance_signed", "mean"),
        )
        .reset_index()
    )

    summary = summary.sort_values("path_importance_mean", ascending=False)

    summary_path = OUT_DIR / "handcrafted_weight_importance_5fold_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    plot_top_features(
        summary,
        score_col="first_layer_l2_mean",
        title="Top Handcrafted Features by First-Layer Weight Norm",
        out_path=OUT_DIR / "figure_handcrafted_first_layer_importance.png",
        top_k=30,
    )

    plot_top_features(
        summary,
        score_col="path_importance_mean",
        title="Top Handcrafted Features by Approximate Path Importance",
        out_path=OUT_DIR / "figure_handcrafted_path_importance.png",
        top_k=30,
    )

    print("\n===== Top 20 handcrafted features by approximate path importance =====")
    print(
        summary[
            [
                "feature_index",
                "feature_name",
                "path_importance_mean",
                "first_layer_l2_mean",
                "signed_path_mean",
            ]
        ]
        .head(20)
        .round(6)
        .to_string(index=False)
    )


def main():
    summarize_across_folds()


if __name__ == "__main__":
    main()
