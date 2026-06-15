from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(
    metadata_csv: str,
    output_csv: str,
    n_folds: int = 5,
    seed: int = 42,
    task: str = "murmur_binary",
):
    meta = pd.read_csv(metadata_csv)

    patient_df = (
        meta[["patient_id", "murmur_label"]]
        .drop_duplicates("patient_id")
        .copy()
    )

    if task == "murmur_binary":
        patient_df = patient_df[
            patient_df["murmur_label"].isin(["Present", "Absent"])
        ].copy()
    else:
        patient_df = patient_df[
            patient_df["murmur_label"].isin(["Present", "Absent", "Unknown"])
        ].copy()

    patient_df = patient_df.reset_index(drop=True)
    patient_df["fold"] = -1

    x = patient_df["patient_id"].values
    y = patient_df["murmur_label"].values
    groups = patient_df["patient_id"].values

    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    for fold, (_, val_idx) in enumerate(splitter.split(x, y, groups)):
        patient_df.loc[val_idx, "fold"] = fold

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Number of patients:", patient_df["patient_id"].nunique())
    print()
    print("Fold distribution:")
    print(patient_df["fold"].value_counts().sort_index())
    print()
    print("Label distribution per fold:")
    print(pd.crosstab(patient_df["fold"], patient_df["murmur_label"]))

    return patient_df
