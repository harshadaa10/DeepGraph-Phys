import pandas as pd

df = pd.read_csv("data/features/ffpp_features.csv")

columns = [
    "video_name",
    "label",
    "split",
    "group_id",
    "source_id",
    "target_id",
    "manipulation",
]

print("=== Videos within each group ===")
for group_id, group in df.sort_values(
    ["group_id", "label", "video_name"]
).groupby("group_id"):
    print(f"\nGroup: {group_id} | Split: {group['split'].iloc[0]} | Videos: {len(group)}")
    print(group[columns].to_string(index=False))

print("\n=== Group size and label composition ===")
summary = (
    df.groupby(["split", "group_id", "label"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
print(summary.to_string(index=False))

print("\nInspection complete. No model was trained and no test metrics were calculated.")