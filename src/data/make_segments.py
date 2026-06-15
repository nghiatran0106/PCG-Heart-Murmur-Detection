from pathlib import Path
import pandas as pd
import soundfile as sf


def get_duration_sec(wav_path: str) -> float:
    info = sf.info(wav_path)
    return info.frames / info.samplerate


def make_segments(
    metadata_csv: str,
    folds_csv: str,
    output_csv: str,
    window_sec: float = 5.0,
    hop_sec: float = 2.5,
    min_valid_sec: float = 1.0,
):
    meta = pd.read_csv(metadata_csv)
    folds = pd.read_csv(folds_csv)

    meta = meta.merge(
        folds[["patient_id", "fold"]],
        on="patient_id",
        how="inner",
    )

    rows = []

    for _, row in meta.iterrows():
        duration = get_duration_sec(row["wav_path"])

        start = 0.0
        idx = 0

        while start < duration:
            remaining = duration - start

            # Bỏ segment cuối nếu quá ngắn
            if remaining < min_valid_sec:
                break

            end = start + window_sec
            segment_id = f"{row['recording_id']}_{idx:04d}"

            rows.append({
                "segment_id": segment_id,
                "patient_id": row["patient_id"],
                "recording_id": row["recording_id"],
                "wav_path": row["wav_path"],
                "location": row["location"],
                "start_sec": start,
                "end_sec": end,
                "duration_sec": duration,
                "fold": row["fold"],
                "label": row["murmur_label"],
            })

            start += hop_sec
            idx += 1

    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Segments:", len(out))
    print("Patients:", out["patient_id"].nunique())
    print(out["label"].value_counts())

    return out
