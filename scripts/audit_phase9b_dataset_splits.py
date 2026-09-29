import pandas as pd

FEATURE_PATH = "data/features/ffpp_features.csv"

df = pd.read_csv(FEATURE_PATH)

required_columns = [
    "video_name",
    "label",
    "split",
    "group_id",
    "source_id",
    "target_id",
    "manipulation",
]

missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("=== Dataset overview ===")
print("Total videos:", len(df))
print("Duplicate video names:", df["video_name"].duplicated().sum())
print("\nVideos by split and label:")
print(pd.crosstab(df["split"], df["label"], margins=True))

print("\n=== Group distribution ===")
print("Unique group IDs:", df["group_id"].nunique())
print("\nUnique groups by split:")
print(df.groupby("split")["group_id"].nunique())

print("\n=== Group overlap between splits ===")
groups_by_split = {
    split: set(part["group_id"].dropna())
    for split, part in df.groupby("split")
}

split_names = list(groups_by_split)
for i, split_a in enumerate(split_names):
    for split_b in split_names[i + 1:]:
        overlap = groups_by_split[split_a] & groups_by_split[split_b]
        print(f"{split_a} vs {split_b}: {len(overlap)} shared groups")
        if overlap:
            print("  Shared group IDs:", sorted(overlap))

print("\n=== Manipulation distribution ===")
print(pd.crosstab(df["split"], df["manipulation"], margins=True))

print("\n=== Source and target ID counts ===")
for column in ["source_id", "target_id"]:
    print(f"\n{column}: {df[column].nunique()} unique non-null values")
    print(pd.crosstab(df["split"], df[column]))

print("\nAudit complete. No model was trained and no test metrics were calculated.")