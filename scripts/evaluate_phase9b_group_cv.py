import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

BASELINE_PATH = "data/features/ffpp_features.csv"
GRAPH_PATH = "data/features/ffpp_graph_features.csv"

# Load feature tables, then restrict modeling data to training rows only.
baseline_all = pd.read_csv(BASELINE_PATH)
graph_all = pd.read_csv(GRAPH_PATH)

train = baseline_all.loc[baseline_all["split"].eq("train")].copy()
if train.empty:
    raise ValueError("No training rows found.")

# Keep graph features aligned to the training videos by video identity.
graph_feature_columns = [
    column for column in graph_all.columns
    if column.startswith("motion_gsp_")
    or column.startswith("physiology_gsp_")
]

graph_train = train[["video_name"]].merge(
    graph_all[["video_name"] + graph_feature_columns],
    on="video_name",
    how="left",
    validate="one_to_one",
)

if graph_train[graph_feature_columns].isna().any().any():
    raise ValueError("Missing graph features after aligning training videos.")

metadata_columns = {
    "video_name", "video_path", "label", "dataset", "manipulation",
    "source_id", "target_id", "group_id", "split",
    "fps", "frames_processed", "rgb_detected_frames",
    "face_detection_rate", "rppg_samples", "rppg_segment_count",
    "shortest_rppg_segment", "longest_rppg_segment",
    "interpolated_frames", "interpolation_rate",
    "motion_samples", "graph_samples",
    "consecutive_graph_transitions", "graph_gap_transitions",
    "largest_graph_frame_difference",
    "synchronized_interpolated_frames",
    "synchronized_interpolation_rate",
    "feature_count", "graph_feature_count",
}

baseline_features = [
    column for column in train.columns
    if column not in metadata_columns
]

feature_sets = {
    "baseline_60": train[baseline_features].reset_index(drop=True),
    "graph_profile_45": graph_train[graph_feature_columns].reset_index(drop=True),
    "combined_105": pd.concat(
        [
            train[baseline_features].reset_index(drop=True),
            graph_train[graph_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    ),
}

labels = train["label"].reset_index(drop=True)
groups = train["group_id"].reset_index(drop=True)

if groups.isna().any():
    raise ValueError("Training rows contain missing group IDs.")

if set(labels.unique()) != {"real", "fake"}:
    raise ValueError(f"Unexpected training labels: {sorted(labels.unique())}")

logo = LeaveOneGroupOut()
results = []

print("=== Phase 9B: Training-only Leave-One-Group-Out CV ===")
print("Training videos:", len(train))
print("Training groups:", groups.nunique())
print("Training real videos:", int(labels.eq("real").sum()))
print("Training fake videos:", int(labels.eq("fake").sum()))
print("Validation/test metrics calculated: 0")
print()

for feature_set_name, features in feature_sets.items():
    X = features.to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values found in {feature_set_name}.")

    out_of_fold_scores = np.full(len(train), np.nan)
    fold_aucs = []

    for fold_number, (train_indices, heldout_indices) in enumerate(
        logo.split(X, labels, groups), start=1
    ):
        train_labels = labels.iloc[train_indices]
        heldout_labels = labels.iloc[heldout_indices]

        # Fit only on genuine videos from the non-held-out groups.
        genuine_train_indices = train_indices[
            train_labels.eq("real").to_numpy()
        ]

        scaler = StandardScaler()
        X_genuine_train = scaler.fit_transform(X[genuine_train_indices])
        X_heldout = scaler.transform(X[heldout_indices])

        model = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.10,
        )
        model.fit(X_genuine_train)

        # Higher score means more anomalous.
        scores = -model.decision_function(X_heldout)
        out_of_fold_scores[heldout_indices] = scores

        y_heldout = heldout_labels.map({"real": 0, "fake": 1}).to_numpy()
        fold_auc = roc_auc_score(y_heldout, scores)
        fold_aucs.append(fold_auc)

        heldout_group = groups.iloc[heldout_indices].iloc[0]
        print(
            f"{feature_set_name} | fold {fold_number} | "
            f"group {heldout_group} | "
            f"genuine training videos {len(genuine_train_indices)} | "
            f"held-out videos {len(heldout_indices)} | "
            f"fold ROC-AUC {fold_auc:.4f}"
        )

    if np.isnan(out_of_fold_scores).any():
        raise ValueError(f"Some rows were not scored for {feature_set_name}.")

    y_all = labels.map({"real": 0, "fake": 1}).to_numpy()
    pooled_auc = roc_auc_score(y_all, out_of_fold_scores)

    results.append({
        "feature_set": feature_set_name,
        "feature_count": features.shape[1],
        "pooled_group_cv_roc_auc": pooled_auc,
        "mean_fold_roc_auc": float(np.mean(fold_aucs)),
        "std_fold_roc_auc": float(np.std(fold_aucs)),
    })

    print()

print("=== Pooled out-of-fold comparison ===")
print(pd.DataFrame(results).to_string(index=False))

print()
print("Interpretation cautions:")
print("- Only the 24 training videos were used for these CV folds.")
print("- Each fold holds out one complete group.")
print("- Fold AUCs are coarse because each held-out group has only 2 real and 2 fake videos.")
print("- Pooled ROC-AUC is not classification accuracy.")
print("- This small-sample result is exploratory, not a final generalization claim.")
print("- The reserved test split was not evaluated.")