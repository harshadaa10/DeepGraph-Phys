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

# Align graph features to the baseline video order.
graph_aligned = baseline[["video_name"]].merge(
    graph[["video_name"] + graph_features],
    on="video_name",
    how="left",
    validate="one_to_one",
)

if graph_aligned[graph_features].isna().any().any():
    raise ValueError("Missing graph features after video alignment.")

# Use the frozen train/test split.
train_mask = baseline["split"].eq("train")
test_mask = baseline["split"].eq("test")

train_labels = baseline.loc[train_mask, "label"].reset_index(drop=True)
test_labels = baseline.loc[test_mask, "label"].reset_index(drop=True)

# Only genuine training examples are used for fitting.
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

print("=== Phase 10C: Reserved Test Evaluation ===")
print()
print("Training videos:", int(train_mask.sum()))
print("Genuine training videos:", int(genuine_train_mask.sum()))
print("Test videos:", int(test_mask.sum()))
print("Test real videos:", int(test_labels.eq("real").sum()))
print("Test fake videos:", int(test_labels.eq("fake").sum()))
print()

results = []

for feature_set_name, all_features in feature_sets.items():

    X_train = all_features.loc[train_mask].reset_index(drop=True)
    X_test = all_features.loc[test_mask].reset_index(drop=True)

    X_genuine_train = X_train.loc[genuine_train_mask].to_numpy()
    X_test = X_test.to_numpy()

    if not pd.notna(X_genuine_train).all():
        raise ValueError(
            f"Non-finite values in genuine training features: "
            f"{feature_set_name}"
        )

    if not pd.notna(X_test).all():
        raise ValueError(
            f"Non-finite values in test features: "
            f"{feature_set_name}"
        )

    # Fit scaling parameters ONLY on genuine training videos.
    scaler = StandardScaler()
    X_genuine_train_scaled = scaler.fit_transform(X_genuine_train)

    # Test data is transformed but never used to fit the scaler.
    X_test_scaled = scaler.transform(X_test)

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )

    # Fit ONLY on genuine training videos.
    model.fit(X_genuine_train_scaled)

    # Higher score means more anomalous.
    test_scores = -model.decision_function(X_test_scaled)

    # real=0, fake=1
    y_test = test_labels.map(
        {"real": 0, "fake": 1}
    ).to_numpy()

    test_auc = roc_auc_score(y_test, test_scores)

    results.append(
        {
            "feature_set": feature_set_name,
            "feature_count": X_genuine_train.shape[1],
            "test_videos": len(test_labels),
            "test_real": int(test_labels.eq("real").sum()),
            "test_fake": int(test_labels.eq("fake").sum()),
            "test_roc_auc": float(test_auc),
        }
    )

    print(
        f"{feature_set_name}: "
        f"Test ROC-AUC = {test_auc:.4f}"
    )

results_df = pd.DataFrame(results)

output_path = "outputs/experiments/phase10c_test_results.csv"
results_df.to_csv(output_path, index=False)

print()
print("Results saved to:", output_path)
print()
print(results_df.to_string(index=False))