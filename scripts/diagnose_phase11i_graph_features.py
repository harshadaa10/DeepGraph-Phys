from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "phase11d_expanded_features.csv"


METADATA_COLUMNS = {
    "video",
    "label",
    "split",
    "group_id",
    "identity_a",
    "identity_b",
    "source_path",
}


def main():
    print("=" * 70)
    print("PHASE 11I - GRAPH FEATURE DIAGNOSIS")
    print("=" * 70)

    df = pd.read_csv(FEATURE_PATH)

    # IMPORTANT:
    # Only training data are used.
    train = df[df["split"] == "train"].copy()

    print(f"\nTotal dataset rows: {len(df)}")
    print(f"Training rows used: {len(train)}")
    print("Validation/test rows used: 0")

    # Identify the 45 graph-profile features.
    # The graph profile contains:
    # - 35 motion GSP features
    # - 10 physiology GSP features
    graph_features = [
    col for col in df.columns
    if col.startswith("motion_gsp_")
    or col.startswith("physiology_gsp_")
    ]

    print(f"\nGraph features found: {len(graph_features)}")

    if len(graph_features) != 45:
        raise ValueError(
            f"Expected 45 graph features, found {len(graph_features)}"
        )

    # Check numeric / finite values.
    numeric_check = train[graph_features].apply(
        pd.to_numeric, errors="coerce"
    )

    non_numeric = numeric_check.isna().sum().sum()
    infinite_values = np.isinf(numeric_check.to_numpy()).sum()

    print(f"Non-numeric/missing values: {non_numeric}")
    print(f"Infinite values: {infinite_values}")

    # Basic statistics.
    stats = pd.DataFrame({
        "feature": graph_features,
        "mean": train[graph_features].mean().values,
        "std": train[graph_features].std().values,
        "min": train[graph_features].min().values,
        "max": train[graph_features].max().values,
        "unique_values": [
            train[col].nunique()
            for col in graph_features
        ],
    })

    # Real/fake means.
    real = train[train["label"] == "real"]
    fake = train[train["label"] == "fake"]

    stats["real_mean"] = [
        real[col].mean() for col in graph_features
    ]

    stats["fake_mean"] = [
        fake[col].mean() for col in graph_features
    ]

    stats["mean_difference_fake_minus_real"] = (
        stats["fake_mean"] - stats["real_mean"]
    )

    # Standardized mean difference.
    pooled_std = np.sqrt(
        (
            real[graph_features].var(ddof=1).values
            + fake[graph_features].var(ddof=1).values
        ) / 2
    )

    stats["standardized_difference"] = np.divide(
        stats["mean_difference_fake_minus_real"].values,
        pooled_std,
        out=np.zeros(len(pooled_std)),
        where=pooled_std != 0,
    )

    # Sort by absolute standardized difference.
    stats["abs_standardized_difference"] = (
        stats["standardized_difference"].abs()
    )

    stats = stats.sort_values(
        "abs_standardized_difference",
        ascending=False
    )

    print("\n" + "-" * 70)
    print("TOP GRAPH FEATURES BY ABSOLUTE STANDARDIZED DIFFERENCE")
    print("-" * 70)

    print(
        stats[
            [
                "feature",
                "real_mean",
                "fake_mean",
                "mean_difference_fake_minus_real",
                "standardized_difference",
                "std",
            ]
        ].head(15).to_string(index=False)
    )

    print("\n" + "-" * 70)
    print("ALL 45 GRAPH FEATURE NAMES")
    print("-" * 70)

    for i, feature in enumerate(graph_features, start=1):
        print(f"{i:02d}. {feature}")

    # Save diagnostic table.
    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "experiments"
        / "phase11i_graph_feature_diagnostics.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("PHASE 11I CHECKPOINT 1 COMPLETE")
    print("=" * 70)
    print(f"Diagnostic rows: {len(stats)}")
    print(f"Saved to: {output_path}")
    print("Validation/test data were not used.")


if __name__ == "__main__":
    main()