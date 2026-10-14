from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = Path("data/features/phase11d_expanded_features.csv")
OUTPUT_DIR = Path("outputs/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "validation"].copy()

# Use the same eligible baseline feature definition as before.
baseline_features = [
    c for c in df.columns
    if (
        c.startswith("motion_") or c.startswith("physiology_")
    )
    and "_gsp_" not in c
    and c != "motion_samples"
]

motion_features = [
    c for c in baseline_features
    if c.startswith("motion_")
]

top5_features = [
    "motion_spectral_centroid_std",
    "motion_low_ratio_std",
    "motion_low_ratio_mean",
    "motion_spectral_centroid_mean",
    "motion_spectral_centroid_max",
]

feature_sets = {
    "baseline_60": baseline_features,
    "motion_only": motion_features,
    "top_5_motion": top5_features,
}

# Labels: real = 0, fake = 1.
y_train = (train["label"] == "fake").astype(int)
y_val = (val["label"] == "fake").astype(int)

results = []
predictions = []

for name, features in feature_sets.items():
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing features: {missing}")

    X_train = train[features].apply(pd.to_numeric, errors="coerce")
    X_val = val[features].apply(pd.to_numeric, errors="coerce")

    # Fit imputation values on training data only.
    medians = X_train.median().fillna(0)
    X_train = X_train.fillna(medians).fillna(0)
    X_val = X_val.fillna(medians).fillna(0)

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_val)[:, 1]
    predicted = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predicted, labels=[0, 1]
    ).ravel()

    results.append({
        "model": name,
        "feature_count": len(features),
        "roc_auc": roc_auc_score(y_val, scores),
        "pr_auc": average_precision_score(y_val, scores),
        "accuracy": accuracy_score(y_val, predicted),
        "precision": precision_score(
            y_val, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_val, predicted, zero_division=0
        ),
        "f1": f1_score(y_val, predicted, zero_division=0),
        "real_correct": int(tn),
        "real_flagged_fake": int(fp),
        "fake_missed": int(fn),
        "fake_detected": int(tp),
    })

    for i, (_, row) in enumerate(val.iterrows()):
        predictions.append({
            "model": name,
            "video": row["video"],
            "group_id": row["group_id"],
            "actual_label": row["label"],
            "predicted_label": "fake" if predicted[i] else "real",
            "fake_score": scores[i],
        })

summary = pd.DataFrame(results)
prediction_df = pd.DataFrame(predictions)

summary.to_csv(
    OUTPUT_DIR / "phase22_motion_ablation_summary.csv",
    index=False,
)
prediction_df.to_csv(
    OUTPUT_DIR / "phase22_motion_ablation_predictions.csv",
    index=False,
)

print("Training videos:", len(train))
print("Validation videos:", len(val))
print("Motion features:", len(motion_features))
print("\n=== VALIDATION COMPARISON ===")
print(summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.3f}",
))
print("\nSaved summary:")
print(OUTPUT_DIR / "phase22_motion_ablation_summary.csv")
print("\nSaved predictions:")
print(OUTPUT_DIR / "phase22_motion_ablation_predictions.csv")
