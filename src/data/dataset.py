import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.preprocessing.filters import (
    resample_if_needed,
    bandpass_filter,
    normalize_audio,
    pad_or_crop,
)
from src.features.mfcc import extract_mfcc_features
from src.features.spectral import extract_spectral_features


class CirCorSegmentDataset(Dataset):
    def __init__(
        self,
        segments_csv: str,
        fold: int,
        mode: str,
        cfg: dict,
        mel_extractor=None,
    ):
        self.df = pd.read_csv(segments_csv)

        if mode == "train":
            self.df = self.df[self.df["fold"] != fold].reset_index(drop=True)
        elif mode == "val":
            self.df = self.df[self.df["fold"] == fold].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.cfg = cfg
        self.mode = mode
        self.mel_extractor = mel_extractor

        if cfg["data"]["task"] == "murmur_binary":
            self.label_map = {
                "Absent": 0,
                "Present": 1,
            }
            self.df = self.df[
                self.df["label"].isin(self.label_map.keys())
            ].reset_index(drop=True)
        else:
            self.label_map = {
                "Absent": 0,
                "Present": 1,
                "Unknown": 2,
            }

        self.target_sr = cfg["preprocessing"]["target_sr"]
        self.window_sec = cfg["segmentation"]["window_sec"]
        self.target_len = int(self.target_sr * self.window_sec)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x, sr = sf.read(row["wav_path"])

        if x.ndim > 1:
            x = x.mean(axis=1)

        start = int(row["start_sec"] * sr)
        end = int(row["end_sec"] * sr)
        x = x[start:end]

        x, sr = resample_if_needed(x, sr, self.target_sr)

        x = pad_or_crop(x, self.target_len)

        x = bandpass_filter(
            x,
            sr=sr,
            low=self.cfg["preprocessing"]["bandpass_low"],
            high=self.cfg["preprocessing"]["bandpass_high"],
            order=self.cfg["preprocessing"]["filter_order"],
        )

        x = normalize_audio(
            x,
            method=self.cfg["preprocessing"]["normalize"],
        )

        x = pad_or_crop(x, self.target_len)

        mfcc_feat = extract_mfcc_features(
            x,
            sr=sr,
            n_mfcc=self.cfg["features"]["n_mfcc"],
        )

        spectral_feat = extract_spectral_features(x, sr=sr)

        handcrafted = torch.tensor(
            list(mfcc_feat) + list(spectral_feat),
            dtype=torch.float32,
        )

        waveform = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        if self.mel_extractor is not None:
            logmel = self.mel_extractor(waveform)
        else:
            logmel = waveform

        label = self.label_map[row["label"]]

        return {
            "logmel": logmel.float(),
            "handcrafted": handcrafted.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": str(row["patient_id"]),
            "recording_id": str(row["recording_id"]),
            "segment_id": str(row["segment_id"]),
        }
