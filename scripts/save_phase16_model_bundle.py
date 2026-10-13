from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_phase13c_cmpc_validation import get_existing_feature_columns

FEATURES_PATH = ROOT / "data" / "features" / "phase11d_expanded_features.csv"
THRESHOLD_PATH = ROOT / "outputs" / "experiments" / "phase16_threshold_results.csv"
BUNDLE_PATH = ROOT / "models" / "phase16_baseline_60.joblib"


def main():
    # Keep only training rows for fitting. Do not evaluate the reserved test set.
    df = pd.read_csv(FEATURES_PATH)
    train_df = df.loc[df["split"].eq("train")].copy()

    feature_columns, graph_features = get_existing_feature_columns(df)
    if len(feature_columns) != 60:
        raise ValueError(
            f"Expected exactly 60 baseline features, got {len(feature_columns)}."
        )

    genuine_train = train_df.loc[train_df["label"].eq("real")].copy()
    if len(genuine_train) != 34:
        raise ValueError(
            f"Expected 34 genuine training videos, got {len(genuine_train)}."
        )

    X_train = genuine_train[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(X_train).all():
        raise ValueError("Training features contain NaN or infinite values.")

    # Fit preprocessing and model on genuine training videos only.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)
    model.fit(X_scaled)

    # Reuse the threshold already selected using validation data.
    threshold_df = pd.read_csv(THRESHOLD_PATH)
    if len(threshold_df) != 1:
        raise ValueError(
            f"Expected one selected threshold row, got {len(threshold_df)}."
        )
    if "model_name" in threshold_df.columns:
        if threshold_df.loc[0, "model_name"] != "baseline_60":
            raise ValueError("The saved threshold is not for baseline_60.")

    threshold = float(threshold_df.loc[0, "threshold"])

    bundle = {
        "model_name": "baseline_60",
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "score_direction": "higher_is_more_likely_fake",
        "threshold_rule": "anomaly_score >= threshold => fake",
        "training_split": "train",
        "training_rows": int(len(genuine_train)),
        "feature_count": len(feature_columns),
        "threshold_source": "validation",
    }

    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)

    print("PHASE 16 — MODEL BUNDLE SAVED")
    print(f"Model: {bundle['model_name']}")
    print(f"Features: {bundle['feature_count']}")
    print(f"Genuine training videos: {bundle['training_rows']}")
    print(f"Selected validation threshold: {threshold:.8f}")
    print(f"Saved bundle: {BUNDLE_PATH}")
    print("Reserved test set: not evaluated")


if __name__ == "__main__":
    main()


