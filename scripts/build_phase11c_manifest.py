from pathlib import Path
import pandas as pd


# ============================================================
# Phase 11C: Build expanded FaceForensics++ group manifest
# ============================================================

REAL_DIR = Path(
    "data/datasets/FaceForensics++_expanded/"
    "original_sequences/youtube/c23/videos"
)

FAKE_DIR = Path(
    "data/datasets/FaceForensics++_expanded/"
    "manipulated_sequences/Deepfakes/c23/videos"
)

OUTPUT = Path(
    "data/features/phase11c_expanded_manifest.csv"
)


# ------------------------------------------------------------
# 1. Discover identity pairs from fake filenames
# ------------------------------------------------------------

fake_files = sorted(FAKE_DIR.glob("*.mp4"))

pairs = sorted(
    {
        tuple(sorted(path.stem.split("_")))
        for path in fake_files
    }
)


# ------------------------------------------------------------
# 2. Verify and build four videos per group
# ------------------------------------------------------------

rows = []

for group_id, (identity_a, identity_b) in enumerate(pairs, start=1):

    videos = [
        (f"{identity_a}.mp4", "real"),
        (f"{identity_b}.mp4", "real"),
        (f"{identity_a}_{identity_b}.mp4", "fake"),
        (f"{identity_b}_{identity_a}.mp4", "fake"),
    ]

    for filename, label in videos:

        if label == "real":
            source_path = REAL_DIR / filename
        else:
            source_path = FAKE_DIR / filename

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing file: {source_path}"
            )

        rows.append(
            {
                "group_id": f"group_{group_id:02d}",
                "identity_a": identity_a,
                "identity_b": identity_b,
                "video": filename,
                "label": label,
                "source_path": str(source_path),
            }
        )


# ------------------------------------------------------------
# 3. Deterministic group-disjoint split
# ------------------------------------------------------------

# 25 groups total:
#   Groups 01-17 -> train
#   Groups 18-21 -> validation
#   Groups 22-25 -> test

def assign_split(group_number):

    if group_number <= 17:
        return "train"

    if group_number <= 21:
        return "validation"

    return "test"


for row in rows:

    group_number = int(
        row["group_id"].split("_")[1]
    )

    row["split"] = assign_split(group_number)


# ------------------------------------------------------------
# 4. Create dataframe
# ------------------------------------------------------------

df = pd.DataFrame(rows)


# ------------------------------------------------------------
# 5. Save manifest
# ------------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)


# ------------------------------------------------------------
# 6. Print audit summary
# ------------------------------------------------------------

print("=" * 60)
print("Phase 11C Expanded Dataset Manifest")
print("=" * 60)

print(f"Total groups: {df['group_id'].nunique()}")
print(f"Total videos: {len(df)}")

print("\nVideos by split:")
print(
    df.groupby("split")["video"]
    .count()
    .to_string()
)

print("\nReal/Fake by split:")
print(
    pd.crosstab(
        df["split"],
        df["label"]
    ).to_string()
)

print("\nGroups by split:")
print(
    df.groupby("split")["group_id"]
    .nunique()
    .to_string()
)

print("\nIdentity groups:")
for group_id, group_df in df.groupby("group_id"):
    identities = (
        group_df["identity_a"].iloc[0],
        group_df["identity_b"].iloc[0],
    )

    split = group_df["split"].iloc[0]

    print(
        f"{group_id}: "
        f"{identities[0]} <-> {identities[1]} "
        f"-> {split}"
    )

print("\nManifest saved to:")
print(OUTPUT)

print("\nSTATUS: PASS")