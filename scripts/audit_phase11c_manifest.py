from pathlib import Path
import pandas as pd


MANIFEST = Path(
    "data/features/phase11c_expanded_manifest.csv"
)


df = pd.read_csv(MANIFEST)


print("=" * 60)
print("Phase 11C Manifest Audit")
print("=" * 60)

# ------------------------------------------------------------
# Basic counts
# ------------------------------------------------------------

print(f"Rows: {len(df)}")
print(f"Unique videos: {df['video'].nunique()}")
print(f"Unique groups: {df['group_id'].nunique()}")


# ------------------------------------------------------------
# Duplicate checks
# ------------------------------------------------------------

duplicate_videos = df[df["video"].duplicated(keep=False)]

print(
    f"\nDuplicate video names: "
    f"{duplicate_videos['video'].nunique()}"
)


# ------------------------------------------------------------
# Split counts
# ------------------------------------------------------------

print("\nSplit counts:")
print(df["split"].value_counts().sort_index())


# ------------------------------------------------------------
# Label counts
# ------------------------------------------------------------

print("\nLabel counts by split:")
print(
    pd.crosstab(
        df["split"],
        df["label"]
    ).sort_index()
)


# ------------------------------------------------------------
# Group overlap
# ------------------------------------------------------------

groups_by_split = {
    split: set(group_df["group_id"])
    for split, group_df in df.groupby("split")
}

train_groups = groups_by_split["train"]
val_groups = groups_by_split["validation"]
test_groups = groups_by_split["test"]

print("\nGroup overlap:")
print("Train ∩ Validation:", train_groups & val_groups)
print("Train ∩ Test:", train_groups & test_groups)
print("Validation ∩ Test:", val_groups & test_groups)


# ------------------------------------------------------------
# Identity overlap
# ------------------------------------------------------------

identities_by_split = {}

for split, group_df in df.groupby("split"):
    identities = set(group_df["identity_a"]) | set(
        group_df["identity_b"]
    )
    identities_by_split[split] = identities

train_ids = identities_by_split["train"]
val_ids = identities_by_split["validation"]
test_ids = identities_by_split["test"]

print("\nIdentity overlap:")
print("Train ∩ Validation:", train_ids & val_ids)
print("Train ∩ Test:", train_ids & test_ids)
print("Validation ∩ Test:", val_ids & test_ids)


# ------------------------------------------------------------
# File existence
# ------------------------------------------------------------

missing_files = []

for path in df["source_path"]:
    if not Path(path).exists():
        missing_files.append(path)

print(
    f"\nMissing source files: "
    f"{len(missing_files)}"
)


# ------------------------------------------------------------
# Final status
# ------------------------------------------------------------

passed = (
    len(df) == 100
    and df["video"].nunique() == 100
    and df["group_id"].nunique() == 25
    and len(duplicate_videos) == 0
    and not (train_groups & val_groups)
    and not (train_groups & test_groups)
    and not (val_groups & test_groups)
    and not (train_ids & val_ids)
    and not (train_ids & test_ids)
    and not (val_ids & test_ids)
    and len(missing_files) == 0
)

print("\n" + "=" * 60)
print("STATUS:", "PASS" if passed else "FAIL")
print("=" * 60)

if not passed:
    raise SystemExit(1)