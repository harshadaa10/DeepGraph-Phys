from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

DATA_PATH = Path("data/features/phase11d_expanded_features.csv")
OUTPUT_DIR = Path("outputs/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
train = df[df["split"] == "train"].copy().reset_index(drop=True)

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

y = (train["label"] == "fake").astype(int).to_numpy()
groups = train["group_id"].to_numpy()

cv = GroupKFold(n_splits=17)

all_results = []
all_predictions = []

for model_name, features in feature_sets.items():
    missing = [c for c in features if c not in train.columns]
    if missing:
        raise ValueError(f"{model_name}: missing features {missing}")

    X = train[features].apply(pd.to_numeric, errors="coerce")
    oof_scores = np.full(len(train), np.nan)
    oof_predictions = np.full(len(train), -1, dtype=int)
    fold_aucs = []
    fold_pr_aucs = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        X_fold_train = X.iloc[train_idx].copy()
        X_fold_val = X.iloc[val_idx].copy()

        # Fit missing-value replacements on this fold's training data only.
        medians = X_fold_train.median().fillna(0)
        X_fold_train = X_fold_train.fillna(medians).fillna(0)
        X_fold_val = X_fold_val.fillna(medians).fillna(0)

        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_fold_train, y[train_idx])

        scores = model.predict_proba(X_fold_val)[:, 1]
        predicted = (scores >= 0.5).astype(int)

        oof_scores[val_idx] = scores
        oof_predictions[val_idx] = predicted

        fold_aucs.append(
            roc_auc_score(y[val_idx], scores)
        )
        fold_pr_aucs.append(
            average_precision_score(y[val_idx], scores)
        )

        all_predictions.extend([
            {
                "model": model_name,
                "fold": fold,
                "video": train.iloc[i]["video"],
                "group_id": groups[i],
                "actual_label": train.iloc[i]["label"],
                "fake_score": oof_scores[i],
                "predicted_label": (
                    "fake" if oof_predictions[i] == 1 else "real"
                ),
            }
            for i in val_idx
        ])

    all_results.append({
        "model": model_name,
        "feature_count": len(features),
        "pooled_oof_roc_auc": roc_auc_score(y, oof_scores),
        "pooled_oof_pr_auc": average_precision_score(y, oof_scores),
        "pooled_oof_accuracy": accuracy_score(y, oof_predictions),
        "pooled_oof_precision": precision_score(
            y, oof_predictions, zero_division=0
        ),
        "pooled_oof_recall": recall_score(
            y, oof_predictions, zero_division=0
        ),
        "pooled_oof_f1": f1_score(
            y, oof_predictions, zero_division=0
        ),
        "mean_fold_roc_auc": np.mean(fold_aucs),
        "fold_roc_auc_sd": np.std(fold_aucs),
        "mean_fold_pr_auc": np.mean(fold_pr_aucs),
    })

summary = pd.DataFrame(all_results)
predictions = pd.DataFrame(all_predictions)

summary.to_csv(
    OUTPUT_DIR / "phase22_motion_group_cv_summary.csv",
    index=False,
)
predictions.to_csv(
    OUTPUT_DIR / "phase22_motion_group_cv_predictions.csv",
    index=False,
)

print("Training videos:", len(train))
print("Training groups:", train["group_id"].nunique())
print("\n=== GROUP-HELD-OUT RESULTS ===")
print(summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.3f}",
))
print("\nSaved summary:",
      OUTPUT_DIR / "phase22_motion_group_cv_summary.csv")
print("Saved predictions:",
      OUTPUT_DIR / "phase22_motion_group_cv_predictions.csv")
