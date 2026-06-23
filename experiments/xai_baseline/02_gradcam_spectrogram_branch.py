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
    p = Path(f"outputs/checkpoints/best_fold{fold}.pt")
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint: {p}")
    return p


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    fixed = {}
    for k, v in sd.items():
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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, logmel_tensor, hand_tensor, class_idx=1):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(logmel_tensor, hand_tensor)
        probs = F.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        prob_present = float(probs[0, 1].detach().cpu().item())

        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=logmel_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, pred, prob_present


def choose_samples(df, fold, max_per_class):
    val = df[df["fold"] == fold].copy()

    val = val[val["murmur_label"].isin(["Absent", "Present"])].copy()
    val["murmur_target"] = val["murmur_label"].map({"Absent": 0, "Present": 1})

    # Avoid too many segments from the same recording.
    val = val.drop_duplicates("recording_id")

    parts = []
    for label in ["Present", "Absent"]:
        sub = val[val["murmur_label"] == label].head(max_per_class)
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)

    if len(out) == 0:
        raise RuntimeError(f"No valid murmur samples found for fold {fold}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--csv", default="data/processed_outcome_binary/segments_outcome_binary_win5p0.csv")
    parser.add_argument("--max-per-class", type=int, default=4)
    parser.add_argument("--class-idx", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = load_checkpoint(find_checkpoint(args.fold), device)
    hand_dim = detect_hand_dim(state_dict)

    model = FusionResNetClassifier(
        handcrafted_dim=hand_dim,
        num_classes=2,
        dropout=0.3,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    target_layer = model.deep_branch.backbone.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    df = pd.read_csv(args.csv)
    samples = choose_samples(df, args.fold, args.max_per_class)

    out_dir = Path("outputs/xai_baseline/gradcam_spectrogram") / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print("=" * 80)
    print(f"Fold: {args.fold}")
    print(f"Device: {device}")
    print(f"Checkpoint handcrafted_dim: {hand_dim}")
    print(f"Samples: {len(samples)}")
    print(f"Output: {out_dir}")
    print("=" * 80)

    for idx, row in samples.iterrows():
        audio_path = Path(row["processed_pcg_path"])
        if not audio_path.exists():
            audio_path = Path(row["wav_path"])

        if not audio_path.exists():
            print(f"Skip missing file: {audio_path}")
            continue

        logmel = make_logmel(str(audio_path))
        logmel_tensor = torch.from_numpy(logmel)[None, None, :, :].to(device)

        # Explain spectrogram branch only. Handcrafted branch is neutralized.
        hand_tensor = torch.zeros((1, hand_dim), dtype=torch.float32, device=device)

        cam, pred, prob_present = gradcam(
            logmel_tensor,
            hand_tensor,
            class_idx=args.class_idx,
        )

        patient_id = str(row["patient_id"])
        recording_id = str(row["recording_id"])
        true_label = str(row["murmur_label"])
        true_target = int(row["murmur_target"])

        plt.figure(figsize=(8, 4))
        plt.imshow(logmel, aspect="auto", origin="lower")
        plt.imshow(cam, aspect="auto", origin="lower", alpha=0.45)
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")
        plt.title(
            f"Fold {args.fold} | {recording_id} | True={true_label} | "
            f"Pred={pred} | P(Present)={prob_present:.3f}"
        )
        plt.tight_layout()

        out_path = out_dir / f"{idx:03d}_{recording_id}_true_{true_label}_pred_{pred}.png"
        plt.savefig(out_path, dpi=180)
        plt.close()

        summary_rows.append(
            {
                "fold": args.fold,
                "patient_id": patient_id,
                "recording_id": recording_id,
                "true_label": true_label,
                "true_target": true_target,
                "pred": pred,
                "prob_present": prob_present,
                "audio_path": str(audio_path),
                "gradcam_path": str(out_path),
            }
        )

        print(f"Saved: {out_path}")

    gradcam.remove()

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"fold{args.fold}_gradcam_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved summary:", summary_path)


if __name__ == "__main__":
    main()
