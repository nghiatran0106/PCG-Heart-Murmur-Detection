import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
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
from torch.utils.data import DataLoader, Dataset

try:
    import librosa
except ImportError as exc:
    raise ImportError(
        "librosa is required for log-Mel and handcrafted feature extraction. "
        "Install it with: pip install librosa"
    ) from exc

from src.models.resnet_fusion import FusionResNetClassifier


@dataclass
class BaselineTrainingConfig:
    project_root: str = "."
    segments_csv: str = "data/processed_outcome_binary/segments_outcome_binary_win5p0.csv"
    output_dir: str = "outputs"

    task: str = "outcome_binary"
    fold: int = 0
    num_classes: int = 2

    sample_rate: int = 4000
    n_mels: int = 128
    n_fft: int = 512
    hop_length: int = 128
    max_frames: int = 157

    handcrafted_dim: int = 32

    epochs: int = 40
    batch_size: int = 32
    num_workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    seed: int = 42

    amp: bool = True
    device: str = "auto"

    monitor_metric: str = "f1"
    save_best_only: bool = True

    max_train_segments: int = 0
    max_val_segments: int = 0


class YAMLConfigLoader:
    @staticmethod
    def load(path: Optional[str], fold: int) -> BaselineTrainingConfig:
        cfg = BaselineTrainingConfig(fold=fold)

        if path is None:
            return cfg

        path_obj = Path(path)
        if not path_obj.exists():
            print(f"[WARN] Config file not found: {path}. Using default config.")
            return cfg

        with open(path_obj, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        flat = YAMLConfigLoader._flatten(raw)

        mapping = {
            "data.segments_csv": "segments_csv",
            "data.processed_segments_csv": "segments_csv",
            "data.outcome_segments_csv": "segments_csv",
            "project.output_dir": "output_dir",
            "training.epochs": "epochs",
            "training.batch_size": "batch_size",
            "training.num_workers": "num_workers",
            "training.lr": "lr",
            "training.learning_rate": "lr",
            "training.weight_decay": "weight_decay",
            "training.dropout": "dropout",
            "project.seed": "seed",
            "model.dropout": "dropout",
            "model.num_classes": "num_classes",
            "features.n_mels": "n_mels",
            "features.n_fft": "n_fft",
            "features.hop_length": "hop_length",
            "features.max_frames": "max_frames",
            "features.handcrafted_dim": "handcrafted_dim",
            "data.task": "task",
            "task": "task",
        }

        for source_key, target_key in mapping.items():
            if source_key in flat:
                setattr(cfg, target_key, flat[source_key])

        cfg.fold = fold

        if cfg.task == "murmur_3class":
            cfg.num_classes = 3
        elif cfg.task in {"outcome_binary", "murmur_binary"}:
            cfg.num_classes = 2

        return cfg

    @staticmethod
    def _flatten(obj: Dict, prefix: str = "") -> Dict:
        out = {}
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(YAMLConfigLoader._flatten(v, key))
            else:
                out[key] = v
        return out


class ReproducibilityManager:
    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


class LabelEncoder:
    def __init__(self, task: str):
        self.task = task

        if task == "outcome_binary":
            self.label_col = "outcome_label"
            self.target_col = "outcome_target"
            self.mapping = {"Normal": 0, "Abnormal": 1, "normal": 0, "abnormal": 1}
        elif task == "murmur_binary":
            self.label_col = "murmur_label"
            self.target_col = "murmur_target"
            self.mapping = {"Absent": 0, "Present": 1, "absent": 0, "present": 1}
        elif task == "murmur_3class":
            self.label_col = "murmur_label"
            self.target_col = "murmur_target"
            self.mapping = {
                "Absent": 0,
                "Present": 1,
                "Unknown": 2,
                "absent": 0,
                "present": 1,
                "unknown": 2,
            }
        else:
            raise ValueError(f"Unsupported task: {task}")

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.task == "murmur_binary" and self.label_col in df.columns:
            df = df[df[self.label_col].isin(["Absent", "Present", "absent", "present"])].copy()
        return df

    def encode_row(self, row: pd.Series) -> int:
        if self.target_col in row and not pd.isna(row[self.target_col]):
            return int(row[self.target_col])

        if self.label_col not in row:
            raise KeyError(
                f"Cannot find label column '{self.label_col}' or target column "
                f"'{self.target_col}' in segments CSV."
            )

        label = row[self.label_col]
        if label not in self.mapping:
            raise ValueError(f"Unknown label '{label}' for task '{self.task}'.")

        return int(self.mapping[label])


class AudioIO:
    def __init__(self, project_root: Path, target_sr: int):
        self.project_root = project_root
        self.target_sr = target_sr

    def read_audio(self, path_value: str) -> np.ndarray:
        path = Path(str(path_value))
        if not path.is_absolute():
            path = self.project_root / path

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)

        y = y.astype(np.float32)

        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)

        if len(y) == 0:
            y = np.zeros(self.target_sr, dtype=np.float32)

        return y


class LogMelExtractor:
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        max_frames: int,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames

    def extract(self, y: np.ndarray) -> torch.Tensor:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)

        logmel = self._normalize(logmel)
        logmel = self._pad_or_crop(logmel)

        return torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        mean = float(np.mean(x))
        std = float(np.std(x))
        if std < 1e-8:
            std = 1.0
        return (x - mean) / std

    def _pad_or_crop(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] > self.max_frames:
            return x[:, : self.max_frames]

        if x.shape[1] < self.max_frames:
            pad_width = self.max_frames - x.shape[1]
            return np.pad(x, ((0, 0), (0, pad_width)), mode="constant")

        return x


class HandcraftedPCGFeatureExtractor:
    def __init__(self, sample_rate: int, n_fft: int = 512, hop_length: int = 128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_dim = 32

    def extract(self, y: np.ndarray) -> torch.Tensor:
        y = np.asarray(y, dtype=np.float32)

        if len(y) < 8:
            return torch.zeros(self.feature_dim, dtype=torch.float32)

        y = self._safe_normalize(y)

        features: List[float] = []
        features.extend(self._time_features(y))
        features.extend(self._spectral_features(y))
        features.extend(self._mfcc_features(y))

        arr = np.asarray(features, dtype=np.float32)

        if len(arr) < self.feature_dim:
            arr = np.pad(arr, (0, self.feature_dim - len(arr)), mode="constant")
        elif len(arr) > self.feature_dim:
            arr = arr[: self.feature_dim]

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(arr, dtype=torch.float32)

    @staticmethod
    def _safe_normalize(y: np.ndarray) -> np.ndarray:
        y = y - np.mean(y)
        std = np.std(y)
        if std < 1e-8:
            std = 1.0
        return y / std

    def _time_features(self, y: np.ndarray) -> List[float]:
        abs_y = np.abs(y)
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        return [
            float(np.mean(y)),
            float(np.std(y)),
            float(np.sqrt(np.mean(y ** 2))),
            float(np.max(abs_y)),
            float(np.ptp(y)),
            float(np.median(abs_y)),
            float(np.percentile(y, 25)),
            float(np.percentile(y, 75)),
            float(np.percentile(y, 75) - np.percentile(y, 25)),
            float(np.mean(zcr)),
        ]

    def _spectral_features(self, y: np.ndarray) -> List[float]:
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        return [
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.mean(bandwidth)),
            float(np.std(bandwidth)),
            float(np.mean(rolloff)),
            float(np.std(rolloff)),
            float(np.mean(flatness)),
            float(np.std(flatness)),
        ]

    def _mfcc_features(self, y: np.ndarray) -> List[float]:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=7,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return list(np.mean(mfcc, axis=1).astype(float)) + list(np.std(mfcc, axis=1).astype(float))


class BaselineFusionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        project_root: Path,
        label_encoder: LabelEncoder,
        audio_io: AudioIO,
        logmel_extractor: LogMelExtractor,
        handcrafted_extractor: HandcraftedPCGFeatureExtractor,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.project_root = project_root
        self.label_encoder = label_encoder
        self.audio_io = audio_io
        self.logmel_extractor = logmel_extractor
        self.handcrafted_extractor = handcrafted_extractor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]

        tf_path = self._get_audio_path(row, preferred="processed_tf_path")
        pcg_path = self._get_audio_path(row, preferred="processed_pcg_path")

        tf_audio = self.audio_io.read_audio(tf_path)
        pcg_audio = self.audio_io.read_audio(pcg_path)

        logmel = self.logmel_extractor.extract(tf_audio)
        handcrafted = self.handcrafted_extractor.extract(pcg_audio)
        target = self.label_encoder.encode_row(row)

        patient_id = str(row["patient_id"]) if "patient_id" in row else str(row.get("recording_id", index))
        recording_id = str(row["recording_id"]) if "recording_id" in row else str(index)

        return {
            "logmel": logmel,
            "handcrafted": handcrafted,
            "target": torch.tensor(target, dtype=torch.long),
            "patient_id": patient_id,
            "recording_id": recording_id,
        }

    @staticmethod
    def _get_audio_path(row: pd.Series, preferred: str) -> str:
        candidates = [preferred, "wav_path", "processed_path", "path"]
        for col in candidates:
            if col in row and not pd.isna(row[col]):
                return str(row[col])
        raise KeyError(f"Cannot find any audio path column from candidates: {candidates}")


class DataModule:
    def __init__(self, cfg: BaselineTrainingConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root).resolve()
        self.label_encoder = LabelEncoder(cfg.task)

        self.audio_io = AudioIO(self.project_root, cfg.sample_rate)
        self.logmel_extractor = LogMelExtractor(
            sample_rate=cfg.sample_rate,
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            max_frames=cfg.max_frames,
        )
        self.handcrafted_extractor = HandcraftedPCGFeatureExtractor(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )

    def setup(self) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
        segments_path = self.project_root / self.cfg.segments_csv
        if not segments_path.exists():
            raise FileNotFoundError(f"Segments CSV not found: {segments_path}")

        df = pd.read_csv(segments_path)
        df = self.label_encoder.filter_dataframe(df)

        if "fold" not in df.columns:
            raise KeyError("segments CSV must contain a 'fold' column.")

        train_df = df[df["fold"] != self.cfg.fold].copy()
        val_df = df[df["fold"] == self.cfg.fold].copy()

        if self.cfg.max_train_segments and self.cfg.max_train_segments > 0:
            train_df = train_df.sample(
                n=min(self.cfg.max_train_segments, len(train_df)),
                random_state=self.cfg.seed,
            ).copy()

        if self.cfg.max_val_segments and self.cfg.max_val_segments > 0:
            val_df = val_df.sample(
                n=min(self.cfg.max_val_segments, len(val_df)),
                random_state=self.cfg.seed,
            ).copy()

        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(
                f"Empty train/val split for fold {self.cfg.fold}. "
                f"Train={len(train_df)}, Val={len(val_df)}"
            )

        train_ds = BaselineFusionDataset(
            dataframe=train_df,
            project_root=self.project_root,
            label_encoder=self.label_encoder,
            audio_io=self.audio_io,
            logmel_extractor=self.logmel_extractor,
            handcrafted_extractor=self.handcrafted_extractor,
        )
        val_ds = BaselineFusionDataset(
            dataframe=val_df,
            project_root=self.project_root,
            label_encoder=self.label_encoder,
            audio_io=self.audio_io,
            logmel_extractor=self.logmel_extractor,
            handcrafted_extractor=self.handcrafted_extractor,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, train_df, val_df


class PatientLevelAggregator:
    @staticmethod
    def aggregate(
        probabilities: np.ndarray,
        targets: np.ndarray,
        patient_ids: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df = pd.DataFrame({"patient_id": patient_ids, "target": targets})

        prob_cols = []
        for i in range(probabilities.shape[1]):
            col = f"prob_{i}"
            df[col] = probabilities[:, i]
            prob_cols.append(col)

        grouped = df.groupby("patient_id", sort=False)

        agg_probs = grouped[prob_cols].mean().values
        agg_targets = grouped["target"].first().values.astype(int)
        agg_patient_ids = list(grouped.groups.keys())

        return agg_probs, agg_targets, agg_patient_ids


class MetricComputer:
    @staticmethod
    def compute_binary(targets: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        pred = np.argmax(probs, axis=1)
        pos_prob = probs[:, 1]

        cm = confusion_matrix(targets, pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp + 1e-8)
        else:
            specificity = float("nan")

        metrics = {
            "accuracy": accuracy_score(targets, pred),
            "precision": precision_score(targets, pred, zero_division=0),
            "recall_sensitivity": recall_score(targets, pred, zero_division=0),
            "specificity": specificity,
            "f1": f1_score(targets, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(targets, pred),
            "mcc": matthews_corrcoef(targets, pred),
            "kappa": cohen_kappa_score(targets, pred),
        }

        try:
            metrics["auroc"] = roc_auc_score(targets, pos_prob)
        except Exception:
            metrics["auroc"] = float("nan")

        try:
            metrics["average_precision"] = average_precision_score(targets, pos_prob)
        except Exception:
            metrics["average_precision"] = float("nan")

        return {k: float(v) for k, v in metrics.items()}

    @staticmethod
    def compute_multiclass(targets: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        pred = np.argmax(probs, axis=1)

        metrics = {
            "accuracy": accuracy_score(targets, pred),
            "macro_f1": f1_score(targets, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(targets, pred, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(targets, pred),
            "mcc": matthews_corrcoef(targets, pred),
            "kappa": cohen_kappa_score(targets, pred),
        }

        try:
            metrics["auroc_ovr"] = roc_auc_score(targets, probs, multi_class="ovr")
        except Exception:
            metrics["auroc_ovr"] = float("nan")

        return {k: float(v) for k, v in metrics.items()}


class BaselineTrainer:
    def __init__(self, cfg: BaselineTrainingConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root).resolve()
        self.output_dir = self.project_root / cfg.output_dir
        self.device = self._resolve_device(cfg.device)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.report_dir = self.output_dir / "reports"
        self.prediction_dir = self.output_dir / "predictions"
        self.log_dir = self.output_dir / "logs"

        for path in [self.checkpoint_dir, self.report_dir, self.prediction_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.model = FusionResNetClassifier(
            handcrafted_dim=cfg.handcrafted_dim,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
        ).to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and self.device.type == "cuda"))

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, train_df: pd.DataFrame) -> None:
        criterion = nn.CrossEntropyLoss(weight=self._build_class_weights(train_df))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.cfg.epochs),
        )

        best_score = -math.inf
        history = []

        print("=" * 80)
        print("TRAIN BASELINE FUSION RESNET")
        print("=" * 80)
        print("Config:", json.dumps(asdict(self.cfg), indent=2))
        print("Device:", self.device)
        print("Train batches:", len(train_loader))
        print("Val batches:", len(val_loader))
        print("Trainable parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print("=" * 80)

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, criterion, optimizer)
            val_metrics, val_predictions = self.evaluate(val_loader)

            scheduler.step()

            score = self._select_score(val_metrics)
            row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            history.append(row)

            print(
                f"Epoch {epoch:03d} | "
                f"loss {train_loss:.4f} | "
                f"score {score:.4f} | "
                + " | ".join(
                    f"{k} {v:.4f}"
                    for k, v in val_metrics.items()
                    if isinstance(v, float) and not math.isnan(v)
                )
            )

            is_best = score > best_score
            if is_best:
                best_score = score
                self._save_checkpoint(epoch, score, optimizer)

            self._save_json(history, self.log_dir / f"baseline_fusion_fold{self.cfg.fold}_history.json")
            self._save_json(val_metrics, self.report_dir / f"baseline_fusion_fold{self.cfg.fold}_metrics.json")
            val_predictions.to_csv(
                self.prediction_dir / f"baseline_fusion_fold{self.cfg.fold}_patient_predictions.csv",
                index=False,
            )

        print("=" * 80)
        print(f"Best {self.cfg.monitor_metric}: {best_score:.4f}")
        print("=" * 80)

    def _train_one_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0

        for batch in loader:
            logmel = batch["logmel"].to(self.device, non_blocking=True)
            handcrafted = batch["handcrafted"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.cfg.amp and self.device.type == "cuda")):
                logits = self.model(logmel, handcrafted)
                loss = criterion(logits, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            batch_size = target.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

        return total_loss / max(total_items, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[Dict[str, float], pd.DataFrame]:
        self.model.eval()

        all_probs = []
        all_targets = []
        all_patient_ids = []
        all_recording_ids = []

        for batch in loader:
            logmel = batch["logmel"].to(self.device, non_blocking=True)
            handcrafted = batch["handcrafted"].to(self.device, non_blocking=True)
            target = batch["target"].cpu().numpy()

            logits = self.model(logmel, handcrafted)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(target)
            all_patient_ids.extend(batch["patient_id"])
            all_recording_ids.extend(batch["recording_id"])

        probs = np.concatenate(all_probs, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        patient_probs, patient_targets, patient_ids = PatientLevelAggregator.aggregate(
            probabilities=probs,
            targets=targets,
            patient_ids=all_patient_ids,
        )

        if self.cfg.num_classes == 2:
            metrics = MetricComputer.compute_binary(patient_targets, patient_probs)
        else:
            metrics = MetricComputer.compute_multiclass(patient_targets, patient_probs)

        pred_df = pd.DataFrame({
            "patient_id": patient_ids,
            "target": patient_targets,
            "prediction": np.argmax(patient_probs, axis=1),
        })

        for i in range(patient_probs.shape[1]):
            pred_df[f"prob_{i}"] = patient_probs[:, i]

        return metrics, pred_df

    def _build_class_weights(self, train_df: pd.DataFrame) -> Optional[torch.Tensor]:
        label_encoder = LabelEncoder(self.cfg.task)
        targets = train_df.apply(label_encoder.encode_row, axis=1).values.astype(int)

        counts = np.bincount(targets, minlength=self.cfg.num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (self.cfg.num_classes * counts)

        print("Class counts:", counts.tolist())
        print("Class weights:", weights.tolist())

        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _select_score(self, metrics: Dict[str, float]) -> float:
        if self.cfg.monitor_metric in metrics:
            value = metrics[self.cfg.monitor_metric]
        elif "f1" in metrics:
            value = metrics["f1"]
        elif "macro_f1" in metrics:
            value = metrics["macro_f1"]
        else:
            value = metrics.get("accuracy", 0.0)

        if math.isnan(value):
            return -math.inf
        return float(value)

    def _save_checkpoint(
        self,
        epoch: int,
        score: float,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        path = self.checkpoint_dir / f"baseline_fusion_fold{self.cfg.fold}_best.pt"

        torch.save(
            {
                "epoch": epoch,
                "score": score,
                "config": asdict(self.cfg),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )
        print(f"Saved best checkpoint: {path}")

    @staticmethod
    def _save_json(obj, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)


class BaselineExperimentRunner:
    def __init__(self, cfg: BaselineTrainingConfig):
        self.cfg = cfg

    def run(self) -> None:
        ReproducibilityManager.set_seed(self.cfg.seed)

        data_module = DataModule(self.cfg)
        train_loader, val_loader, train_df, val_df = data_module.setup()

        print("Train segments:", len(train_df))
        print("Val segments:", len(val_df))
        if "patient_id" in train_df.columns:
            print("Train patients:", train_df["patient_id"].nunique())
            print("Val patients:", val_df["patient_id"].nunique())

        trainer = BaselineTrainer(self.cfg)
        trainer.fit(train_loader, val_loader, train_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OOP baseline FusionResNet model for PCG classification.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index.")
    parser.add_argument("--task", type=str, default=None, choices=["outcome_binary", "murmur_binary", "murmur_3class"])
    parser.add_argument("--segments-csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-segments", type=int, default=None)
    parser.add_argument("--max-val-segments", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = YAMLConfigLoader.load(args.config, args.fold)

    if args.task is not None:
        cfg.task = args.task
        cfg.num_classes = 3 if args.task == "murmur_3class" else 2

    if args.segments_csv is not None:
        cfg.segments_csv = args.segments_csv
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.lr is not None:
        cfg.lr = args.lr
    if args.amp:
        cfg.amp = True
    if args.no_amp:
        cfg.amp = False
    if args.max_train_segments is not None:
        cfg.max_train_segments = args.max_train_segments
    if args.max_val_segments is not None:
        cfg.max_val_segments = args.max_val_segments

    runner = BaselineExperimentRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
