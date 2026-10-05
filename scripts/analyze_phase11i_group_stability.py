from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "phase11d_expanded_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "phase11i_group_stability.csv"
)


def main():
    print("=" * 70)
    print("PHASE 11I - GRAPH FEATURE GROUP STABILITY")
    print("=" * 70)

    df = pd.read_csv(FEATURE_PATH)

    # Training data only.
    train = df[df["split"] == "train"].copy()

    print(f"\nTotal dataset rows: {len(df)}")
    print(f"Training rows used: {len(train)}")
    print("Validation/test rows used: 0")

    graph_features = [
        col
        for col in df.columns
        if col.startswith("motion_gsp_")
        or col.startswith("physiology_gsp_")
    ]

    if len(graph_features) != 45:
        raise ValueError(
            f"Expected 45 graph features, found {len(graph_features)}"
        )

    groups = sorted(train["group_id"].unique())

    print(f"Training groups: {len(groups)}")

    results = []

    for feature in graph_features:

        group_differences = []

        for group in groups:

            group_df = train[train["group_id"] == group]

            real_values = group_df.loc[
                group_df["label"] == "real",
                feature
            ]

            fake_values = group_df.loc[
                group_df["label"] == "fake",
                feature
            ]

            if len(real_values) != 2 or len(fake_values) != 2:
                raise ValueError(
                    f"Group {group} does not contain exactly "
                    f"2 real and 2 fake videos."
                )

            real_mean = real_values.mean()
            fake_mean = fake_values.mean()

            difference = fake_mean - real_mean

            group_differences.append(difference)

        differences = np.asarray(group_differences)

        positive_groups = int(np.sum(differences > 0))
        negative_groups = int(np.sum(differences < 0))
        zero_groups = int(np.sum(differences == 0))

        # Consistency = proportion of groups following the dominant direction.
        dominant_count = max(
            positive_groups,
            negative_groups
        )

        consistency = dominant_count / len(groups)

        results.append({
            "feature": feature,
            "mean_group_difference": differences.mean(),
            "median_group_difference": np.median(differences),
            "std_group_difference": differences.std(ddof=1),
            "positive_groups": positive_groups,
            "negative_groups": negative_groups,
            "zero_groups": zero_groups,
            "direction_consistency": consistency,
            "absolute_mean_group_difference": abs(
                differences.mean()
            ),
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        [
            "direction_consistency",
            "absolute_mean_group_difference",
        ],
        ascending=[False, False],
    )

    print("\n" + "-" * 70)
    print("MOST CONSISTENT GRAPH FEATURES ACROSS TRAINING GROUPS")
    print("-" * 70)

    print(
        results_df[
            [
                "feature",
                "mean_group_difference",
                "median_group_difference",
                "std_group_difference",
                "positive_groups",
                "negative_groups",
                "direction_consistency",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "-" * 70)
    print("FEATURES WITH CONSISTENCY >= 70%")
    print("-" * 70)

    consistent = results_df[
        results_df["direction_consistency"] >= 0.70
    ]

    print(f"Count: {len(consistent)}")

    if len(consistent) > 0:
        print(
            consistent[
                [
                    "feature",
                    "direction_consistency",
                    "mean_group_difference",
                    "median_group_difference",
                ]
            ].to_string(index=False)
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("PHASE 11I CHECKPOINT 2 COMPLETE")
    print("=" * 70)
    print(f"Features analyzed: {len(results_df)}")
    print(f"Training groups analyzed: {len(groups)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("Validation/test data were not used.")


if __name__ == "__main__":
    main()