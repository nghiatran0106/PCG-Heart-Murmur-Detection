import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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


def detect_dims(state_dict):
    hand_input_dim = int(state_dict["hand_branch.0.weight"].shape[1])
    deep_dim = int(state_dict["deep_branch.backbone.fc.weight"].shape[0])
    classifier_input_dim = int(state_dict["classifier.0.weight"].shape[0])
    hand_embed_dim = classifier_input_dim - deep_dim
    return hand_input_dim, deep_dim, hand_embed_dim, classifier_input_dim


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


@torch.no_grad()
def forward_deep_only(model, logmel_tensor, hand_embed_dim, class_idx=1):
    model.eval()

    deep_feat = model.deep_branch(logmel_tensor)
    hand_feat = torch.zeros(
        (deep_feat.shape[0], hand_embed_dim),
        dtype=deep_feat.dtype,
        device=deep_feat.device,
    )

    fused = torch.cat([deep_feat, hand_feat], dim=1)
    logits = model.classifier(fused)
    prob = F.softmax(logits, dim=1)[:, class_idx]
    pred = torch.argmax(logits, dim=1)

    return prob, pred


@torch.no_grad()
def occlusion_map(model, logmel, device, hand_embed_dim, patch_mel=16, patch_time=16, class_idx=1):
    x = torch.from_numpy(logmel)[None, None, :, :].to(device)

    base_prob, base_pred = forward_deep_only(
        model=model,
        logmel_tensor=x,
        hand_embed_dim=hand_embed_dim,
        class_idx=class_idx,
    )

    base_prob = float(base_prob.item())
    base_pred = int(base_pred.item())

    n_mel, n_time = logmel.shape
    heat = np.zeros((n_mel, n_time), dtype=np.float32)
    count = np.zeros((n_mel, n_time), dtype=np.float32)

    occluded_tensors = []
    boxes = []

    for m0 in range(0, n_mel, patch_mel):
        for t0 in range(0, n_time, patch_time):
            m1 = min(m0 + patch_mel, n_mel)
            t1 = min(t0 + patch_time, n_time)

            occ = logmel.copy()
            occ[m0:m1, t0:t1] = 0.0

            occluded_tensors.append(torch.from_numpy(occ)[None, None, :, :])
            boxes.append((m0, m1, t0, t1))

    batch = torch.cat(occluded_tensors, dim=0).to(device)

    probs, _ = forward_deep_only(
        model=model,
        logmel_tensor=batch,
        hand_embed_dim=hand_embed_dim,
        class_idx=class_idx,
    )

    probs = probs.detach().cpu().numpy()

    for prob, (m0, m1, t0, t1) in zip(probs, boxes):
        drop = base_prob - float(prob)
        heat[m0:m1, t0:t1] += drop
        count[m0:m1, t0:t1] += 1.0

    heat = heat / (count + 1e-8)
    heat = np.maximum(heat, 0.0)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    return heat, base_prob, base_pred


def choose_samples(df, fold, max_per_class):
    val = df[df["fold"] == fold].copy()
    val = val[val["murmur_label"].isin(["Absent", "Present"])].copy()
    val["murmur_target"] = val["murmur_label"].map({"Absent": 0, "Present": 1})

    val = val.drop_duplicates("recording_id")

    parts = []
    for label in ["Present", "Absent"]:
        sub = val[val["murmur_label"] == label].head(max_per_class)
        parts.append(sub)

    samples = pd.concat(parts, ignore_index=True)

    if len(samples) == 0:
        raise RuntimeError(f"No samples found for fold {fold}")

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--csv", default="data/processed_outcome_binary/segments_outcome_binary_win5p0.csv")
    parser.add_argument("--max-per-class", type=int, default=3)
    parser.add_argument("--patch-mel", type=int, default=16)
    parser.add_argument("--patch-time", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = load_state_dict(find_checkpoint(args.fold), device)
    hand_input_dim, deep_dim, hand_embed_dim, classifier_input_dim = detect_dims(state_dict)

    model = FusionResNetClassifier(
        handcrafted_dim=hand_input_dim,
        num_classes=2,
        dropout=0.3,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    df = pd.read_csv(args.csv)
    samples = choose_samples(df, args.fold, args.max_per_class)

    out_dir = Path("outputs/xai_baseline/occlusion_sensitivity") / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print("=" * 80)
    print(f"Fold: {args.fold}")
    print(f"Device: {device}")
    print(f"Handcrafted input dim from checkpoint: {hand_input_dim}")
    print(f"Deep embedding dim: {deep_dim}")
    print(f"Hand embedding dim: {hand_embed_dim}")
    print(f"Classifier input dim: {classifier_input_dim}")
    print(f"Samples: {len(samples)}")
    print("=" * 80)

    for idx, row in samples.iterrows():
        audio_path = Path(row["processed_pcg_path"])
        if not audio_path.exists():
            audio_path = Path(row["wav_path"])

        if not audio_path.exists():
            print(f"Skip missing audio: {audio_path}")
            continue

        logmel = make_logmel(str(audio_path))

        heat, base_prob, base_pred = occlusion_map(
            model=model,
            logmel=logmel,
            device=device,
            hand_embed_dim=hand_embed_dim,
            patch_mel=args.patch_mel,
            patch_time=args.patch_time,
            class_idx=1,
        )

        recording_id = str(row["recording_id"])
        true_label = str(row["murmur_label"])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(logmel, aspect="auto", origin="lower")
        axes[0].set_title("Log-Mel Spectrogram")
        axes[0].set_xlabel("Time frames")
        axes[0].set_ylabel("Mel bins")

        axes[1].imshow(logmel, aspect="auto", origin="lower")
        axes[1].imshow(heat, aspect="auto", origin="lower", alpha=0.55)
        axes[1].set_title("Occlusion Sensitivity")
        axes[1].set_xlabel("Time frames")
        axes[1].set_ylabel("Mel bins")

        fig.suptitle(
            f"{recording_id} | True={true_label} | Pred={base_pred} | P(Present)={base_prob:.3f}",
            fontsize=11,
        )

        fig.tight_layout()

        out_path = out_dir / f"{idx:03d}_{recording_id}_true_{true_label}_pred_{base_pred}_occlusion.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        summary_rows.append(
            {
                "fold": args.fold,
                "recording_id": recording_id,
                "patient_id": str(row["patient_id"]),
                "true_label": true_label,
                "pred": base_pred,
                "prob_present": base_prob,
                "audio_path": str(audio_path),
                "figure_path": str(out_path),
            }
        )

        print("Saved:", out_path)

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"fold{args.fold}_occlusion_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSaved summary:", summary_path)


if __name__ == "__main__":
    main()
