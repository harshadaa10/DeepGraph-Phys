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
    / "phase11i_feature_redundancy.csv"
)


def main():
    print("=" * 70)
    print("PHASE 11I - GRAPH FEATURE REDUNDANCY ANALYSIS")
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

    X = train[graph_features].astype(float)

    # ---------------------------------------------------------------
    # 1. Pairwise absolute correlation
    # ---------------------------------------------------------------

    corr = X.corr().abs()

    pairs = []

    for i in range(len(graph_features)):
        for j in range(i + 1, len(graph_features)):
            f1 = graph_features[i]
            f2 = graph_features[j]

            pairs.append({
                "feature_1": f1,
                "feature_2": f2,
                "absolute_correlation": corr.loc[f1, f2],
            })

    pair_df = pd.DataFrame(pairs)

    pair_df = pair_df.sort_values(
        "absolute_correlation",
        ascending=False,
    )

    print("\n" + "-" * 70)
    print("TOP 20 MOST REDUNDANT FEATURE PAIRS")
    print("-" * 70)

    print(
        pair_df.head(20).to_string(index=False)
    )

    # ---------------------------------------------------------------
    # 2. Count highly correlated pairs
    # ---------------------------------------------------------------

    thresholds = [0.90, 0.95, 0.99]

    print("\n" + "-" * 70)
    print("HIGH-CORRELATION PAIR COUNTS")
    print("-" * 70)

    for threshold in thresholds:
        count = int(
            (pair_df["absolute_correlation"] >= threshold).sum()
        )

        print(
            f"|correlation| >= {threshold:.2f}: {count} pairs"
        )

    # ---------------------------------------------------------------
    # 3. Correlation by GSP component
    # ---------------------------------------------------------------

    def component(feature):
        if "physiology_gsp_" in feature:
            prefix = "physiology"
        else:
            prefix = "motion"

        for g in range(7):
            if f"_g{g}_" in feature:
                return f"{prefix}_g{g}"

        return "unknown"

    pair_df["component_1"] = pair_df["feature_1"].map(component)
    pair_df["component_2"] = pair_df["feature_2"].map(component)

    within_component = pair_df[
        pair_df["component_1"] == pair_df["component_2"]
    ]

    print("\n" + "-" * 70)
    print("WITHIN-COMPONENT REDUNDANCY")
    print("-" * 70)

    component_summary = []

    components = sorted(
        set(component(f) for f in graph_features)
    )

    for comp in components:
        subset = within_component[
            within_component["component_1"] == comp
        ]

        component_summary.append({
            "component": comp,
            "feature_count": sum(
                component(f) == comp
                for f in graph_features
            ),
            "pair_count": len(subset),
            "mean_absolute_correlation": (
                subset["absolute_correlation"].mean()
                if len(subset) > 0
                else np.nan
            ),
            "max_absolute_correlation": (
                subset["absolute_correlation"].max()
                if len(subset) > 0
                else np.nan
            ),
        })

    component_df = pd.DataFrame(component_summary)

    print(
        component_df.to_string(index=False)
    )

    # ---------------------------------------------------------------
    # 4. Highly correlated feature count per feature
    # ---------------------------------------------------------------

    high_corr_threshold = 0.90

    high_corr_count = {}

    for feature in graph_features:
        count = int(
            (
                corr.loc[feature].drop(feature)
                >= high_corr_threshold
            ).sum()
        )

        high_corr_count[feature] = count

    feature_redundancy = pd.DataFrame({
        "feature": graph_features,
        "high_correlation_partner_count": [
            high_corr_count[f]
            for f in graph_features
        ],
    })

    feature_redundancy = feature_redundancy.sort_values(
        "high_correlation_partner_count",
        ascending=False,
    )

    print("\n" + "-" * 70)
    print("FEATURES WITH MOST |CORRELATION| >= 0.90 PARTNERS")
    print("-" * 70)

    print(
        feature_redundancy.head(15).to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Save pair-level results
    # ---------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pair_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("PHASE 11I CHECKPOINT 3 COMPLETE")
    print("=" * 70)

    print(f"Graph features analyzed: {len(graph_features)}")
    print(f"Feature pairs analyzed: {len(pair_df)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("Validation/test data were not used.")


if __name__ == "__main__":
    main()