from pathlib import Path
import pandas as pd


def normalize_col_name(col: str) -> str:
    return (
        col.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_metadata(raw_dir: str, training_dir: str, metadata_csv: str, output_csv: str) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    training_dir = Path(training_dir)

    df = pd.read_csv(metadata_csv)
    df.columns = [normalize_col_name(c) for c in df.columns]

    # Tìm cột patient id
    possible_patient_cols = ["patient_id", "id"]
    patient_col = None
    for c in possible_patient_cols:
        if c in df.columns:
            patient_col = c
            break

    if patient_col is None:
        raise ValueError(f"Cannot find patient id column. Columns: {df.columns.tolist()}")

    rows = []

    for _, row in df.iterrows():
        patient_id = str(row[patient_col])

        wav_files = sorted(training_dir.glob(f"{patient_id}_*.wav"))

        if len(wav_files) == 0:
            continue

        for wav_path in wav_files:
            recording_id = wav_path.stem
            location = recording_id.split("_")[-1]

            tsv_path = wav_path.with_suffix(".tsv")
            txt_path = training_dir / f"{patient_id}.txt"

            rows.append({
                "patient_id": patient_id,
                "recording_id": recording_id,
                "wav_path": str(wav_path),
                "tsv_path": str(tsv_path) if tsv_path.exists() else "",
                "txt_path": str(txt_path) if txt_path.exists() else "",
                "location": location,
                "murmur_label": row.get("murmur", ""),
                "outcome_label": row.get("outcome", ""),
                "age": row.get("age", ""),
                "sex": row.get("sex", ""),
                "height": row.get("height", ""),
                "weight": row.get("weight", ""),
                "pregnancy_status": row.get("pregnancy_status", ""),
            })

    out = pd.DataFrame(rows)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Rows:", len(out))
    print("Patients:", out["patient_id"].nunique())
    print(out["murmur_label"].value_counts(dropna=False))

    return out
