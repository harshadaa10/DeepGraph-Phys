from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_manifest_splits.csv"
)


# Each tuple represents one connected FaceForensics++ pair.
#
# Example:
# 033.mp4
# 097.mp4
# 033_097.mp4
# 097_033.mp4
#
# All four must remain in the SAME split.
PAIR_GROUPS = [
    ("033", "097"),
    ("183", "253"),
    ("210", "241"),
    ("252", "266"),
    ("339", "392"),
    ("469", "481"),
    ("585", "599"),
    ("672", "720"),
    ("828", "830"),
    ("866", "878"),
]


# Development split:
# 6 groups train
# 2 groups validation
# 2 groups test
GROUP_SPLITS = {
    1: "train",
    2: "train",
    3: "train",
    4: "train",
    5: "train",
    6: "train",
    7: "validation",
    8: "validation",
    9: "test",
    10: "test",
}


def main():
    print(
        "\nDeepGraph-Phys — "
        "FaceForensics++ Leakage-Safe Splits"
    )

    print("-" * 65)

    dataframe = pd.read_csv(
        MANIFEST_PATH,
        dtype={
            "source_id": str,
            "target_id": str,
        },
        keep_default_na=False,
    )

    # Preserve three-digit IDs.
    dataframe["source_id"] = (
        dataframe["source_id"]
        .astype(str)
        .str.zfill(3)
    )

    dataframe["target_id"] = (
        dataframe["target_id"]
        .astype(str)
        .apply(
            lambda x:
            x.zfill(3)
            if x != ""
            else ""
        )
    )

    # Build ID -> group mapping.
    id_to_group = {}

    for group_number, pair in enumerate(
        PAIR_GROUPS,
        start=1,
    ):
        for video_id in pair:
            id_to_group[video_id] = group_number

    group_values = []
    split_values = []

    for _, row in dataframe.iterrows():

        source_id = row["source_id"]
        target_id = row["target_id"]

        relevant_ids = {source_id}

        if target_id:
            relevant_ids.add(target_id)

        matched_groups = {
            id_to_group[video_id]
            for video_id in relevant_ids
            if video_id in id_to_group
        }

        if len(matched_groups) != 1:
            raise ValueError(
                f"Could not assign exactly one group "
                f"to {row['video_name']}. "
                f"Matched groups: {matched_groups}"
            )

        group_number = matched_groups.pop()

        group_values.append(
            group_number
        )

        split_values.append(
            GROUP_SPLITS[group_number]
        )

    dataframe["group_id"] = group_values
    dataframe["split"] = split_values

    # ==========================================
    # DISPLAY GROUPS
    # ==========================================

    print("\nPair groups:")

    for group_number, pair in enumerate(
        PAIR_GROUPS,
        start=1,
    ):
        print(
            f"Group {group_number:02d}: "
            f"{pair[0]} <-> {pair[1]} "
            f"-> {GROUP_SPLITS[group_number]}"
        )

    # ==========================================
    # SUMMARY
    # ==========================================

    print("\nSplit counts:")

    split_counts = pd.crosstab(
        dataframe["split"],
        dataframe["label"],
    )

    print(split_counts)

    # ==========================================
    # VALIDATION
    # ==========================================

    print("\nValidation checks:")

    print(
        "Total rows = 40:",
        len(dataframe) == 40,
    )

    print(
        "Exactly 10 groups:",
        dataframe["group_id"].nunique() == 10,
    )

    rows_per_group = (
        dataframe.groupby("group_id")
        .size()
    )

    print(
        "Exactly 4 videos per group:",
        (rows_per_group == 4).all(),
    )

    # Every source/target ID should occur in only one split.
    id_split_map = {}

    leakage_detected = False

    for _, row in dataframe.iterrows():

        ids = {row["source_id"]}

        if row["target_id"]:
            ids.add(row["target_id"])

        for video_id in ids:

            if video_id not in id_split_map:
                id_split_map[video_id] = row["split"]

            elif (
                id_split_map[video_id]
                != row["split"]
            ):
                leakage_detected = True

    print(
        "No source/target ID leakage:",
        not leakage_detected,
    )

    expected_counts = {
        ("train", "real"): 12,
        ("train", "fake"): 12,
        ("validation", "real"): 4,
        ("validation", "fake"): 4,
        ("test", "real"): 4,
        ("test", "fake"): 4,
    }

    counts_correct = True

    for (
        split_name,
        label_name
    ), expected in expected_counts.items():

        actual = len(
            dataframe[
                (dataframe["split"] == split_name)
                & (dataframe["label"] == label_name)
            ]
        )

        if actual != expected:
            counts_correct = False

    print(
        "Expected class/split counts:",
        counts_correct,
    )

    # ==========================================
    # SAVE
    # ==========================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved split manifest to:"
    )

    print(OUTPUT_PATH)

    print(
        "\nSplit manifest created successfully."
    )


if __name__ == "__main__":
    main()