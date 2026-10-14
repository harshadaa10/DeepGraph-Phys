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

all_motion = [
    c for c in train.columns
    if c.startswith("motion_")
    and "_gsp_" not in c
    and c != "motion_samples"
]

without_centroid = [
    c for c in all_motion
    if not c.startswith("motion_spectral_centroid_")
]

feature_sets = {
    "motion_30": all_motion,
    "motion_without_spectral_centroid": without_centroid,
}

y = (train["label"] == "fake").astype(int).to_numpy()
groups = train["group_id"].to_numpy()
cv = GroupKFold(n_splits=17)

results = []
predictions = []

for model_name, features in feature_sets.items():
    X = train[features].apply(pd.to_numeric, errors="coerce")
    scores_all = np.full(len(train), np.nan)
    predicted_all = np.full(len(train), -1, dtype=int)
    fold_auc = []
    fold_pr_auc = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        X_fit = X.iloc[train_idx].copy()
        X_holdout = X.iloc[val_idx].copy()

        # Missing-value handling is learned from the fold's training data.
        medians = X_fit.median().fillna(0)
        X_fit = X_fit.fillna(medians).fillna(0)
        X_holdout = X_holdout.fillna(medians).fillna(0)

        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_fit, y[train_idx])

        fold_scores = model.predict_proba(X_holdout)[:, 1]
        fold_predictions = (fold_scores >= 0.5).astype(int)

        scores_all[val_idx] = fold_scores
        predicted_all[val_idx] = fold_predictions

        fold_auc.append(roc_auc_score(y[val_idx], fold_scores))
        fold_pr_auc.append(
            average_precision_score(y[val_idx], fold_scores)
        )

        for j, idx in enumerate(val_idx):
            predictions.append({
                "model": model_name,
                "fold": fold,
                "video": train.iloc[idx]["video"],
                "group_id": groups[idx],
                "actual_label": train.iloc[idx]["label"],
                "predicted_label": (
                    "fake" if fold_predictions[j] else "real"
                ),
                "fake_score": fold_scores[j],
            })

    results.append({
        "model": model_name,
        "feature_count": len(features),
        "pooled_oof_roc_auc": roc_auc_score(y, scores_all),
        "pooled_oof_pr_auc": average_precision_score(y, scores_all),
        "pooled_oof_accuracy": accuracy_score(y, predicted_all),
        "pooled_oof_precision": precision_score(
            y, predicted_all, zero_division=0
        ),
        "pooled_oof_recall": recall_score(
            y, predicted_all, zero_division=0
        ),
        "pooled_oof_f1": f1_score(y, predicted_all, zero_division=0),
        "mean_fold_roc_auc": np.mean(fold_auc),
        "fold_roc_auc_sd": np.std(fold_auc),
        "mean_fold_pr_auc": np.mean(fold_pr_auc),
    })

summary = pd.DataFrame(results)
summary.to_csv(
    OUTPUT_DIR / "phase22_spectral_centroid_ablation_summary.csv",
    index=False,
)
pd.DataFrame(predictions).to_csv(
    OUTPUT_DIR / "phase22_spectral_centroid_ablation_predictions.csv",
    index=False,
)

print("Training videos:", len(train))
print("Groups:", train["group_id"].nunique())
print("All motion features:", len(all_motion))
print("Without spectral-centroid features:", len(without_centroid))
print("\n=== GROUP-HELD-OUT ABLATION ===")
print(summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.3f}",
))
print("\nSaved summary:",
      OUTPUT_DIR / "phase22_spectral_centroid_ablation_summary.csv")
print("Saved predictions:",
      OUTPUT_DIR / "phase22_spectral_centroid_ablation_predictions.csv")
