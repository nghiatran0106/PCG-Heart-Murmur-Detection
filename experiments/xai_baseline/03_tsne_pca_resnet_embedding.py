import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.models.resnet_fusion import FusionResNetClassifier


def find_checkpoint(fold: int) -> Path:
    path = Path(f"outputs/checkpoints/best_fold{fold}.pt")
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def load_state_dict(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

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
        fixed[k.replace("module.", "")] = v
    return fixed


def detect_hand_dim(state_dict):
    return int(state_dict["hand_branch.0.weight"].shape[1])


def make_logmel(audio_path, sr=4000, n_mels=128, n_fft=512, hop_length=128, max_frames=157):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    if len(y) == 0:
        y = np.zeros(sr * 5, dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)

    if logmel.shape[1] < max_frames:
        pad = max_frames - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)), mode="constant")
    elif logmel.shape[1] > max_frames:
        logmel = logmel[:, :max_frames]

    return logmel.astype(np.float32)


def choose_samples(df, fold, max_per_class):
    val = df[df["fold"] == fold].copy()
    val = val[val["murmur_label"].isin(["Absent", "Present"])].copy()
    val["murmur_target"] = val["murmur_label"].map({"Absent": 0, "Present": 1})

    val = val.drop_duplicates("recording_id")

    parts = []
    for label in ["Absent", "Present"]:
        sub = val[val["murmur_label"] == label].head(max_per_class)
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)

    if len(out) == 0:
        raise RuntimeError(f"No samples found for fold {fold}")

    return out


@torch.no_grad()
def extract_embeddings(model, samples, device, batch_size=32):
    model.eval()

    embeddings = []
    labels = []
    recording_ids = []

    logmel_list = []
    meta_rows = []

    for _, row in samples.iterrows():
        audio_path = Path(row["processed_pcg_path"])
        if not audio_path.exists():
            audio_path = Path(row["wav_path"])

        if not audio_path.exists():
            continue

        logmel = make_logmel(str(audio_path))
        logmel_list.append(torch.from_numpy(logmel)[None, :, :])

        meta_rows.append(row)

    if not logmel_list:
        raise RuntimeError("No valid audio samples were loaded.")

    for start in range(0, len(logmel_list), batch_size):
        batch = torch.stack(logmel_list[start:start + batch_size], dim=0).to(device)
        # batch shape: [B, 1, 128, T]
        emb = model.deep_branch(batch)
        embeddings.append(emb.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    for row in meta_rows:
        labels.append(int(row["murmur_target"]))
        recording_ids.append(str(row["recording_id"]))

    return embeddings, np.asarray(labels), recording_ids


def run_tsne(features):
    n = features.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))

    try:
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=1000,
            random_state=42,
        )
    except TypeError:
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            n_iter=1000,
            random_state=42,
        )

    return reducer.fit_transform(features)


def plot_scatter(points, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    absent = labels == 0
    present = labels == 1

    ax.scatter(points[absent, 0], points[absent, 1], marker="o", alpha=0.75, label="Absent")
    ax.scatter(points[present, 0], points[present, 1], marker="^", alpha=0.75, label="Present")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--csv", default="data/processed_outcome_binary/segments_outcome_binary_win5p0.csv")
    parser.add_argument("--max-per-class", type=int, default=150)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = load_state_dict(find_checkpoint(args.fold), device)
    hand_dim = detect_hand_dim(state_dict)

    model = FusionResNetClassifier(
        handcrafted_dim=hand_dim,
        num_classes=2,
        dropout=0.3,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    df = pd.read_csv(args.csv)
    samples = choose_samples(df, args.fold, args.max_per_class)

    print("=" * 80)
    print(f"Fold: {args.fold}")
    print(f"Device: {device}")
    print(f"Checkpoint handcrafted_dim: {hand_dim}")
    print(f"Samples selected: {len(samples)}")
    print("=" * 80)

    features, labels, recording_ids = extract_embeddings(model, samples, device)

    out_dir = Path("outputs/xai_baseline/tsne_resnet_embedding") / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_points = PCA(n_components=2, random_state=42).fit_transform(features)
    tsne_points = run_tsne(features)

    emb_df = pd.DataFrame(
        {
            "recording_id": recording_ids,
            "label": labels,
            "label_name": ["Present" if x == 1 else "Absent" for x in labels],
            "pca_1": pca_points[:, 0],
            "pca_2": pca_points[:, 1],
            "tsne_1": tsne_points[:, 0],
            "tsne_2": tsne_points[:, 1],
        }
    )

    emb_path = out_dir / f"fold{args.fold}_resnet_embedding_projection.csv"
    emb_df.to_csv(emb_path, index=False)

    plot_scatter(
        pca_points,
        labels,
        "PCA of 256-D ResNet Spectrogram Embeddings",
        out_dir / "figure_pca_resnet_embedding.png",
    )

    plot_scatter(
        tsne_points,
        labels,
        "t-SNE of 256-D ResNet Spectrogram Embeddings",
        out_dir / "figure_tsne_resnet_embedding.png",
    )

    print("Saved:", emb_path)


if __name__ == "__main__":
    main()
