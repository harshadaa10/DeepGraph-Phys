from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "features" / "phase11d_expanded_features.csv"
MODEL_PATH = ROOT / "models" / "phase23_motion_only_rf.joblib"


def main():
    df = pd.read_csv(FEATURES_PATH)

    train = df.loc[df["split"].eq("train")].copy()
    if len(train) != 68:
        raise ValueError(f"Expected 68 training rows; found {len(train)}.")

    features = [
        column for column in df.columns
        if column.startswith("motion_")
        and "_gsp_" not in column
        and column != "motion_samples"
    ]

    if len(features) != 30:
        raise ValueError(f"Expected 30 motion features; found {len(features)}.")

    X_train = train[features].apply(pd.to_numeric, errors="coerce")
    # Fit imputation values on training data only.
    medians = X_train.median().fillna(0)
    X_train = X_train.fillna(medians).fillna(0)
    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain non-finite values.")

    y_train = train["label"].eq("fake").astype(int).to_numpy()

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training data must contain both real and fake labels.")

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    bundle = {
        "model_name": "phase23_motion_only_random_forest",
        "model": model,
        "feature_columns": features,
        "feature_count": len(features),
        "imputation_medians": medians.to_dict(),
        "label_mapping": {"real": 0, "fake": 1},
        "score_type": "predict_proba_fake",
        "decision_threshold": None,
        "threshold_status": "not calibrated; Phase 24 pending",
        "training_split": "train",
        "training_rows": len(train),
        "training_group_count": int(train["group_id"].nunique()),
        "validation_rows_used_for_fitting": 0,
        "random_state": 42,
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "selection_basis": (
            "Phase 21-22 grouped training CV and fixed validation comparison; provisional selection with limited sample size."
            "provisional selection, limited sample size"
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    # Reload to verify the saved artifact and its metadata.
    saved = joblib.load(MODEL_PATH)
    if saved["feature_count"] != 30 or saved["decision_threshold"] is not None:
        raise RuntimeError("Saved bundle verification failed.")

    print("PHASE 23 - PROVISIONAL MODEL BUNDLE SAVED")
    print(f"Model: {saved['model_name']}")
    print(f"Training videos: {saved['training_rows']}")
    print(f"Training groups: {saved['training_group_count']}")
    print(f"Features: {saved['feature_count']}")
    print(f"Validation rows used for fitting: {saved['validation_rows_used_for_fitting']}")
    print(f"Decision threshold: {saved['decision_threshold']} (Phase 24 pending)")
    print(f"Saved and verified: {MODEL_PATH}")
    print("Reserved final-test split: not loaded for evaluation or tuning.")


if __name__ == "__main__":
    main()




