from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


BASELINE_PATH = Path("data/features/phase11d_expanded_features.csv")
CMPC_PATH = Path("data/features/phase13c_cmpc_features.csv")
OUTPUT_PATH = Path("outputs/experiments/phase13c_cmpc_feature_ablation_cv.csv")

BASELINE_FEATURES = [
    "cmpc_forehead_correlation",
    "cmpc_forehead_absolute_correlation",
    "cmpc_left_cheek_correlation",
    "cmpc_left_cheek_absolute_correlation",
    "cmpc_right_cheek_correlation",
    "cmpc_right_cheek_absolute_correlation",
    "cmpc_cross_region_mean_correlation",
    "cmpc_cross_region_std_correlation",
    "cmpc_cross_region_mean_abs_correlation",
    "cmpc_cross_region_min_abs_correlation",
    "cmpc_cross_region_max_abs_correlation",
    "cmpc_mean_motion_std",
    "cmpc_mean_rppg_std",
]

# Load only training videos. Validation and test are not used here.
baseline = pd.read_csv(BASELINE_PATH)
cmpc = pd.read_csv(CMPC_PATH)

train_meta = baseline.loc[
    baseline["split"].eq("train"),
    ["video", "label", "group_id"],
].copy()

cmpc_features = [c for c in cmpc.columns if c.startswith("cmpc_")]

data = train_meta.merge(
    cmpc[["video"] + cmpc_features],
    on="video",
    how="left",
    validate="one_to_one",
)

if len(data) != 68:
    raise ValueError(f"Expected 68 training videos, found {len(data)}")

if data[cmpc_features].isna().any().any():
    raise ValueError("Missing CMPC values detected")

if set(data["label"]) != {"real", "fake"}:
    raise ValueError("Unexpected label values")

# Preserve the same feature definitions used in the previous experiment.
feature_sets = {
    "cmpc_16": cmpc_features,
    "cmpc_13": [c for c in cmpc_features if not c.endswith("_segment_std")],
    "cmpc_12": [
        c for c in cmpc_features
        if not c.endswith("_segment_std")
        and c != "cmpc_mean_rppg_std"
    ],
}

groups = sorted(data["group_id"].unique())
if len(groups) != 17:
    raise ValueError(f"Expected 17 training groups, found {len(groups)}")

rows = []

for feature_set, features in feature_sets.items():
    for held_out_group in groups:
        train = data.loc[data["group_id"] != held_out_group]
        held_out = data.loc[data["group_id"] == held_out_group]

        # Train the anomaly detector only on genuine videos.
        genuine_train = train.loc[train["label"] == "real", features]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(genuine_train)
        x_eval = scaler.transform(held_out[features])

        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)
        model.fit(x_train)

        # Lower One-Class SVM decision values indicate greater anomaly.
        scores = -model.decision_function(x_eval)
        y_true = (held_out["label"] == "fake").astype(int).to_numpy()

        rows.append({
            "feature_set": feature_set,
            "feature_count": len(features),
            "held_out_group": held_out_group,
            "video_count": len(held_out),
            "roc_auc": roc_auc_score(y_true, scores),
            "pr_auc": average_precision_score(y_true, scores),
        })

results = pd.DataFrame(rows)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

summary = (
    results.groupby(["feature_set", "feature_count"])
    .agg(
        folds=("held_out_group", "nunique"),
        mean_roc_auc=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        mean_pr_auc=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", "std"),
    )
    .reset_index()
)

print("TRAINING-ONLY CMPC FEATURE ABLATION")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved fold results to: {OUTPUT_PATH}")
