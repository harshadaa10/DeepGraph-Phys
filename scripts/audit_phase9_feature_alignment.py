import pandas as pd

BASELINE_PATH = "data/features/ffpp_features.csv"
GRAPH_PATH = "data/features/ffpp_graph_features.csv"

baseline = pd.read_csv(BASELINE_PATH)
graph = pd.read_csv(GRAPH_PATH)

identity_columns = [
    "video_name",
    "label",
    "group_id",
    "split",
]

print("BASELINE shape:", baseline.shape)
print("GRAPH shape:", graph.shape)

# Check required identity columns exist.
for column in identity_columns:
    if column not in baseline.columns:
        raise ValueError(f"Missing {column!r} in baseline dataset")
    if column not in graph.columns:
        raise ValueError(f"Missing {column!r} in graph dataset")

# Check duplicate video names.
print("\nDuplicate baseline video names:", baseline["video_name"].duplicated().sum())
print("Duplicate graph video names:", graph["video_name"].duplicated().sum())

# Compare video identities and metadata.
baseline_ids = baseline[identity_columns].sort_values("video_name").reset_index(drop=True)
graph_ids = graph[identity_columns].sort_values("video_name").reset_index(drop=True)

same_rows = baseline_ids.equals(graph_ids)

print("\nIdentity and split metadata match exactly:", same_rows)
print("Baseline video count:", baseline["video_name"].nunique())
print("Graph video count:", graph["video_name"].nunique())

if not same_rows:
    comparison = baseline_ids.merge(
        graph_ids,
        on="video_name",
        how="outer",
        suffixes=("_baseline", "_graph"),
        indicator=True,
    )

    mismatches = comparison[
        (comparison["_merge"] != "both")
        | (comparison["label_baseline"] != comparison["label_graph"])
        | (comparison["group_id_baseline"] != comparison["group_id_graph"])
        | (comparison["split_baseline"] != comparison["split_graph"])
    ]

    print("\nMismatched rows:")
    print(mismatches.to_string(index=False))
else:
    print("\nSplit counts:")
    print(baseline.groupby(["split", "label"]).size().to_string())

# Identify model feature columns.
baseline_metadata = {
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
    "synchronized_interpolation_rate", "feature_count",
}

graph_metadata = {
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
    "synchronized_interpolation_rate", "graph_feature_count",
}

baseline_features = [c for c in baseline.columns if c not in baseline_metadata]
graph_features = [
    c for c in graph.columns
    if c.startswith("motion_gsp_") or c.startswith("physiology_gsp_")
]

print("\nBaseline feature count:", len(baseline_features))
print("Graph profile feature count:", len(graph_features))
print("Feature names overlap:", sorted(set(baseline_features) & set(graph_features)))

print("\nAudit complete.")