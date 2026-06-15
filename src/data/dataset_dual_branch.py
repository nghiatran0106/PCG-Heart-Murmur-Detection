import copy
from pathlib import Path

import pandas as pd

from src.data.dataset import CirCorSegmentDataset


PATH_COLS = [
    "wav_path",
    "audio_path",
    "path",
    "file_path",
    "filepath",
    "wav_file",
]


def infer_stems(df):
    if "recording_id" in df.columns:
        return df["recording_id"].astype(str).str.replace(".wav", "", regex=False)

    for col in PATH_COLS:
        if col in df.columns:
            return df[col].astype(str).apply(lambda p: Path(p).stem)

    raise ValueError(
        "Cannot infer recording stem. Expected recording_id or one of: "
        + ", ".join(PATH_COLS)
    )


def rewrite_audio_paths_in_csv(src_csv, dst_csv, wav_root):
    src_csv = Path(src_csv)
    dst_csv = Path(dst_csv)
    wav_root = Path(wav_root)

    df = pd.read_csv(src_csv)

    stems = infer_stems(df)
    abs_paths = stems.apply(lambda s: str(wav_root / f"{s}.wav"))

    for col in PATH_COLS:
        if col in df.columns:
            df[col] = abs_paths

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_csv, index=False)

    return str(dst_csv)


def make_branch_cfg(cfg, wav_root):
    cfg2 = copy.deepcopy(cfg)
    wav_root = Path(wav_root)

    cfg2.setdefault("data", {})
    cfg2["data"]["training_dir"] = str(wav_root)
    cfg2["data"]["raw_dir"] = str(wav_root.parent)

    return cfg2


class DualBranchCirCorSegmentDataset:
    """
    Dual-branch wrapper đúng với interface dataset gốc:

    Original:
        CirCorSegmentDataset(segments_csv, fold, mode, cfg, mel_extractor)

    Branch PCG:
        bp25_600_zscore wav -> handcrafted features

    Branch TF:
        bp25_600_spectral_zscore wav -> logmel image
    """

    def __init__(self, segments_csv: str, fold: int, mode: str, cfg: dict, mel_extractor=None):
        self.segments_csv = segments_csv
        self.fold = fold
        self.mode = mode
        self.cfg = cfg

        pcg_wav_root = Path(cfg["data"]["pcg_wav_root"])
        tf_wav_root = Path(cfg["data"]["tf_wav_root"])

        tmp_dir = Path(cfg["project"]["output_dir"]) / "_dual_branch_segments"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        pcg_csv = tmp_dir / f"segments_fold{fold}_{mode}_pcg.csv"
        tf_csv = tmp_dir / f"segments_fold{fold}_{mode}_tf.csv"

        pcg_csv = rewrite_audio_paths_in_csv(
            src_csv=segments_csv,
            dst_csv=pcg_csv,
            wav_root=pcg_wav_root,
        )

        tf_csv = rewrite_audio_paths_in_csv(
            src_csv=segments_csv,
            dst_csv=tf_csv,
            wav_root=tf_wav_root,
        )

        pcg_cfg = make_branch_cfg(cfg, pcg_wav_root)
        tf_cfg = make_branch_cfg(cfg, tf_wav_root)

        self.pcg_dataset = CirCorSegmentDataset(
            segments_csv=pcg_csv,
            fold=fold,
            mode=mode,
            cfg=pcg_cfg,
            mel_extractor=mel_extractor,
        )

        self.tf_dataset = CirCorSegmentDataset(
            segments_csv=tf_csv,
            fold=fold,
            mode=mode,
            cfg=tf_cfg,
            mel_extractor=mel_extractor,
        )

        assert len(self.pcg_dataset) == len(self.tf_dataset), (
            len(self.pcg_dataset),
            len(self.tf_dataset),
        )

    def __len__(self):
        return len(self.pcg_dataset)

    def __getitem__(self, idx):
        pcg_item = self.pcg_dataset[idx]
        tf_item = self.tf_dataset[idx]

        item = dict(pcg_item)

        # Handcrafted lấy từ waveform branch.
        item["handcrafted"] = pcg_item["handcrafted"]

        # Log-Mel lấy từ TF-image branch.
        item["logmel"] = tf_item["logmel"]

        return item
