from pathlib import Path
import pandas as pd


ROOT = Path(".")
CIRCOR_DIR = ROOT / "data" / "CirCor2022"
CVD_DIR = ROOT / "data" / "CVD"
PN2016_DIR = ROOT / "data" / "PhysioNet2016"
OUT_DIR = ROOT / "data" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_binary_label(label):
    if pd.isna(label):
        return None

    s = str(label).strip().lower()

    if s in {"absent", "normal", "n", "-1", "0"}:
        return 0

    if s in {"present", "abnormal", "as", "mr", "ms", "mvp", "1"}:
        return 1

    if s in {"unknown", "nan", ""}:
        return None

    return None


def build_circor2022():
    csv_candidates = [
        CIRCOR_DIR / "training_data.csv",
        CIRCOR_DIR / "training_data" / "training_data.csv",
    ]

    metadata_csv = None
    for p in csv_candidates:
        if p.exists():
            metadata_csv = p
            break

    if metadata_csv is None:
        raise FileNotFoundError(
            "Cannot find CirCor2022 training_data.csv. "
            "Expected data/CirCor2022/training_data.csv"
        )

    df = pd.read_csv(metadata_csv)
    df.columns = [c.strip() for c in df.columns]

    patient_col = None
    for c in ["Patient ID", "patient_id", "PatientID"]:
        if c in df.columns:
            patient_col = c
            break

    murmur_col = None
    for c in ["Murmur", "murmur_label", "murmur"]:
        if c in df.columns:
            murmur_col = c
            break

    outcome_col = None
    for c in ["Outcome", "outcome_label", "outcome"]:
        if c in df.columns:
            outcome_col = c
            break

    if patient_col is None:
        raise KeyError(f"Cannot find patient column in {metadata_csv}. Columns={list(df.columns)}")
    if murmur_col is None:
        raise KeyError(f"Cannot find murmur column in {metadata_csv}. Columns={list(df.columns)}")

    meta = {}
    for _, row in df.iterrows():
        pid = str(row[patient_col]).strip()
        meta[pid] = row.to_dict()

    wavs = sorted((CIRCOR_DIR / "training_data").glob("*.wav"))

    rows = []
    for wav in wavs:
        stem = wav.stem

        if "_" in stem:
            patient_id, location = stem.split("_", 1)
        else:
            patient_id, location = stem, "unknown"

        info = meta.get(str(patient_id), {})

        murmur_label = str(info.get(murmur_col, "Unknown")).strip()
        murmur_target = normalize_binary_label(murmur_label)

        outcome_label = str(info.get(outcome_col, "Unknown")).strip() if outcome_col else "Unknown"
        outcome_target = normalize_binary_label(outcome_label)

        rows.append(
            {
                "dataset_name": "CirCor2022",
                "sample_id": f"CirCor2022_{stem}",
                "patient_id": str(patient_id),
                "recording_id": stem,
                "wav_path": str(wav),
                "location": location,
                "original_label": murmur_label,
                "murmur_label": murmur_label,
                "murmur_target": murmur_target,
                "outcome_label": outcome_label,
                "outcome_target": outcome_target,
            }
        )

    out = pd.DataFrame(rows)
    return out


def read_reference_csv(path):
    # PhysioNet 2016 REFERENCE.csv usually has no header: record,label
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Bad REFERENCE.csv: {path}")

    df = df.iloc[:, :2]
    df.columns = ["recording_id", "label"]
    return df


def build_physionet2016():
    rows = []

    for folder in sorted(PN2016_DIR.glob("training-*")):
        if not folder.is_dir():
            continue

        ref = folder / "REFERENCE.csv"
        if not ref.exists():
            print(f"[WARN] Missing REFERENCE.csv in {folder}")
            continue

        ref_df = read_reference_csv(ref)
        label_map = {
            str(r["recording_id"]).strip(): int(r["label"])
            for _, r in ref_df.iterrows()
        }

        for wav in sorted(folder.glob("*.wav")):
            rid = wav.stem
            raw_label = label_map.get(rid, None)

            if raw_label == -1:
                label_name = "Normal"
                target = 0
            elif raw_label == 1:
                label_name = "Abnormal"
                target = 1
            else:
                label_name = "Unknown"
                target = None

            # PhysioNet2016 does not provide explicit murmur labels.
            # We map Normal -> Absent and Abnormal -> Present-like only for external binary comparison.
            murmur_label = "Absent" if target == 0 else ("Present" if target == 1 else "Unknown")
            outcome_label = "Normal" if target == 0 else ("Abnormal" if target == 1 else "Unknown")

            rows.append(
                {
                    "dataset_name": "PhysioNet2016",
                    "sample_id": f"PhysioNet2016_{folder.name}_{rid}",
                    "patient_id": f"{folder.name}_{rid}",
                    "recording_id": rid,
                    "wav_path": str(wav),
                    "location": "unknown",
                    "original_label": label_name,
                    "murmur_label": murmur_label,
                    "murmur_target": target,
                    "outcome_label": outcome_label,
                    "outcome_target": target,
                }
            )

    return pd.DataFrame(rows)


def build_cvd():
    rows = []

    label_folders = {
        "N": {
            "murmur_label": "Absent",
            "murmur_target": 0,
            "outcome_label": "Normal",
            "outcome_target": 0,
        },
        "AS": {
            "murmur_label": "Present",
            "murmur_target": 1,
            "outcome_label": "Abnormal",
            "outcome_target": 1,
        },
        "MR": {
            "murmur_label": "Present",
            "murmur_target": 1,
            "outcome_label": "Abnormal",
            "outcome_target": 1,
        },
        "MS": {
            "murmur_label": "Present",
            "murmur_target": 1,
            "outcome_label": "Abnormal",
            "outcome_target": 1,
        },
        "MVP": {
            "murmur_label": "Present",
            "murmur_target": 1,
            "outcome_label": "Abnormal",
            "outcome_target": 1,
        },
    }

    for label, cfg in label_folders.items():
        folder = CVD_DIR / label
        if not folder.exists():
            print(f"[WARN] Missing CVD folder: {folder}")
            continue

        for wav in sorted(folder.glob("*.wav")):
            rid = wav.stem

            rows.append(
                {
                    "dataset_name": "CVD",
                    "sample_id": f"CVD_{label}_{rid}",
                    "patient_id": f"CVD_{label}_{rid}",
                    "recording_id": rid,
                    "wav_path": str(wav),
                    "location": "unknown",
                    "original_label": label,
                    "murmur_label": cfg["murmur_label"],
                    "murmur_target": cfg["murmur_target"],
                    "outcome_label": cfg["outcome_label"],
                    "outcome_target": cfg["outcome_target"],
                }
            )

    return pd.DataFrame(rows)


def main():
    all_parts = []

    builders = [
        ("CirCor2022", build_circor2022),
        ("PhysioNet2016", build_physionet2016),
        ("CVD", build_cvd),
    ]

    for name, fn in builders:
        print(f"\n===== Building {name} =====")
        df = fn()
        print(df.head())
        print("Rows:", len(df))
        print("Murmur counts:")
        print(df["murmur_label"].value_counts(dropna=False))
        print("Outcome counts:")
        print(df["outcome_label"].value_counts(dropna=False))

        out_path = OUT_DIR / f"{name}_metadata.csv"
        df.to_csv(out_path, index=False)
        print("Saved:", out_path)

        all_parts.append(df)

    full = pd.concat(all_parts, ignore_index=True)

    full["murmur_target"] = pd.to_numeric(full["murmur_target"], errors="coerce")
    full["outcome_target"] = pd.to_numeric(full["outcome_target"], errors="coerce")

    full_path = OUT_DIR / "unified_metadata_all_datasets.csv"
    full.to_csv(full_path, index=False)

    print("\n===== Unified metadata =====")
    print("Saved:", full_path)
    print("Rows:", len(full))
    print("\nBy dataset:")
    print(full["dataset_name"].value_counts())
    print("\nMurmur label counts:")
    print(full.groupby("dataset_name")["murmur_label"].value_counts(dropna=False))
    print("\nOutcome label counts:")
    print(full.groupby("dataset_name")["outcome_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
