import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

BASELINE_PATH = "data/features/ffpp_features.csv"
GRAPH_PATH = "data/features/ffpp_graph_features.csv"

baseline = pd.read_csv(BASELINE_PATH)
graph = pd.read_csv(GRAPH_PATH)

# Metadata is never passed into the model as a feature.
metadata_columns = {
    "video_name",
    "video_path",
    "label",
    "dataset",
    "manipulation",
    "source_id",
    "target_id",
    "group_id",
    "split",
    "fps",
    "frames_processed",
    "rgb_detected_frames",
    "face_detection_rate",
    "rppg_samples",
    "rppg_segment_count",
    "shortest_rppg_segment",
    "longest_rppg_segment",
    "interpolated_frames",
    "interpolation_rate",
    "motion_samples",
    "graph_samples",
    "consecutive_graph_transitions",
    "graph_gap_transitions",
    "largest_graph_frame_difference",
    "synchronized_interpolated_frames",
    "synchronized_interpolation_rate",
    "feature_count",
    "graph_feature_count",
}

baseline_features = [
    column
    for column in baseline.columns
    if column not in metadata_columns
]

graph_features = [
    column
    for column in graph.columns
    if column.startswith("motion_gsp_")
    or column.startswith("physiology_gsp_")
]

# Align the graph feature rows to the baseline video order.
graph_aligned = baseline[["video_name"]].merge(
    graph[["video_name"] + graph_features],
    on="video_name",
    how="left",
    validate="one_to_one",
)

if graph_aligned[graph_features].isna().any().any():
    raise ValueError("Missing graph features after video alignment.")

# Use only training and validation rows.
# The reserved test rows are excluded from the experiment.
train_mask = baseline["split"].eq("train")
validation_mask = baseline["split"].eq("validation")

train_labels = baseline.loc[train_mask, "label"].reset_index(drop=True)
validation_labels = baseline.loc[validation_mask, "label"].reset_index(drop=True)

# Only genuine training examples are used to fit the scaler and model.
genuine_train_mask = train_labels.eq("real")

feature_sets = {
    "baseline_60": baseline[baseline_features].reset_index(drop=True),
    "graph_profile_45": graph_aligned[graph_features].reset_index(drop=True),
    "combined_105": pd.concat(
        [
            baseline[baseline_features].reset_index(drop=True),
            graph_aligned[graph_features].reset_index(drop=True),
        ],
        axis=1,
    ),
}

print("Training videos:", int(train_mask.sum()))
print("Genuine training videos:", int(genuine_train_mask.sum()))
print("Validation videos:", int(validation_mask.sum()))
print("Test rows loaded for evaluation: 0")
print()

results = []

for name, all_features in feature_sets.items():
    X_train = all_features.loc[train_mask].reset_index(drop=True)
    X_validation = all_features.loc[validation_mask].reset_index(drop=True)

    X_genuine_train = X_train.loc[genuine_train_mask].to_numpy()
    X_validation = X_validation.to_numpy()

    if not pd.notna(X_genuine_train).all():
        raise ValueError(f"Non-finite values in genuine training features: {name}")

    if not pd.notna(X_validation).all():
        raise ValueError(f"Non-finite values in validation features: {name}")

    scaler = StandardScaler()
    X_genuine_train_scaled = scaler.fit_transform(X_genuine_train)
    X_validation_scaled = scaler.transform(X_validation)

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )
    model.fit(X_genuine_train_scaled)

    # Larger anomaly scores indicate a sample is less like the genuine
    # training distribution.
    validation_scores = -model.decision_function(X_validation_scaled)

    # fake=1, real=0: higher anomaly score should indicate fake.
    y_validation = validation_labels.map({"real": 0, "fake": 1}).to_numpy()

    auc = roc_auc_score(y_validation, validation_scores)

    results.append(
        {
            "feature_set": name,
            "feature_count": X_genuine_train.shape[1],
            "validation_roc_auc": auc,
            "validation_real_mean_score": validation_scores[y_validation == 0].mean(),
            "validation_fake_mean_score": validation_scores[y_validation == 1].mean(),
        }
    )

results_df = pd.DataFrame(results)

print("Phase 9A — Validation comparison")
print(results_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print()
print("Interpretation note:")
print("- This is a small preliminary validation set (4 real and 4 fake videos).")
print("- ROC-AUC is a ranking measure, not classification accuracy.")
print("- Do not treat small differences as reliable evidence of improvement.")
print("- The reserved test split has not been evaluated.")