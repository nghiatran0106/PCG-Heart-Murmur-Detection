import pandas as pd


def aggregate_segment_to_patient(
    pred_df: pd.DataFrame,
    prob_col: str = "prob_present",
    method: str = "max",
):
    rec_df = (
        pred_df
        .groupby(["patient_id", "recording_id", "label"], as_index=False)[prob_col]
        .mean()
    )

    if method == "mean":
        patient_df = (
            rec_df
            .groupby(["patient_id", "label"], as_index=False)[prob_col]
            .mean()
        )
    elif method == "max":
        patient_df = (
            rec_df
            .groupby(["patient_id", "label"], as_index=False)[prob_col]
            .max()
        )
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return patient_df
