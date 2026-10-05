from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


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
    / "phase11i_reduced_graph_cv_results.csv"
)

CORRELATION_THRESHOLD = 0.90


def get_graph_features(df):
    return [
        col
        for col in df.columns
        if col.startswith("motion_gsp_")
        or col.startswith("physiology_gsp_")
    ]


def select_reduced_features(X_train, feature_names):
    """
    Greedy correlation filtering.

    Selection is performed ONLY on the current CV training fold.
    Features are processed in their original deterministic order.
    If a feature is highly correlated with an already-selected feature,
    it is removed.
    """
    corr = X_train[feature_names].corr().abs()

    selected = []

    for feature in feature_names:
        keep = True

        for previous in selected:
            if corr.loc[feature, previous] >= CORRELATION_THRESHOLD:
                keep = False
                break

        if keep:
            selected.append(feature)

    return selected


def main():
    print("=" * 70)
    print("PHASE 11I - LEAKAGE-SAFE REDUCED GRAPH CROSS-VALIDATION")
    print("=" * 70)

    df = pd.read_csv(FEATURE_PATH)

    train_df = df[df["split"] == "train"].copy()

    graph_features = get_graph_features(df)

    if len(graph_features) != 45:
        raise ValueError(
            f"Expected 45 graph features, found {len(graph_features)}"
        )

    groups = sorted(train_df["group_id"].unique())

    print(f"\nDataset rows: {len(df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Training groups: {len(groups)}")
    print(f"Original graph features: {len(graph_features)}")
    print(f"Correlation threshold: {CORRELATION_THRESHOLD}")
    print("Validation/test data were not used.")

    results = []

    for fold_number, held_out_group in enumerate(groups, start=1):

        fold_train = train_df[
            train_df["group_id"] != held_out_group
        ].copy()

        fold_test = train_df[
            train_df["group_id"] == held_out_group
        ].copy()

        # ----------------------------------------------------------
        # Feature selection happens ONLY inside this fold's
        # training data.
        # ----------------------------------------------------------

        selected_features = select_reduced_features(
            fold_train,
            graph_features,
        )

        removed_count = len(graph_features) - len(selected_features)

        # Genuine-only training.
        genuine_train = fold_train[
            fold_train["label"] == "real"
        ].copy()

        X_train = genuine_train[selected_features].to_numpy()
        X_test = fold_test[selected_features].to_numpy()

        y_test = (
            fold_test["label"] == "fake"
        ).astype(int).to_numpy()

        # Same scaling protocol as Phase 11H.
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Same frozen OCSVM protocol.
        model = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.10,
        )

        model.fit(X_train_scaled)

        anomaly_scores = -model.decision_function(
            X_test_scaled
        )

        roc_auc = roc_auc_score(
            y_test,
            anomaly_scores,
        )

        pr_auc = average_precision_score(
            y_test,
            anomaly_scores,
        )

        results.append({
            "fold": fold_number,
            "held_out_group": held_out_group,
            "original_features": len(graph_features),
            "selected_features": len(selected_features),
            "removed_features": removed_count,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "selected_feature_names": "|".join(
                selected_features
            ),
        })

        print(
            f"\nFold {fold_number:02d} - {held_out_group}"
        )
        print(
            f"Selected features: "
            f"{len(selected_features)}/{len(graph_features)}"
        )
        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )
        print(
            f"PR-AUC: {pr_auc:.4f}"
        )

    results_df = pd.DataFrame(results)

    print("\n" + "-" * 70)
    print("REDUCED GRAPH CV SUMMARY")
    print("-" * 70)

    print(
        f"Mean selected features: "
        f"{results_df['selected_features'].mean():.2f}"
    )

    print(
        f"Selected feature range: "
        f"{results_df['selected_features'].min()} - "
        f"{results_df['selected_features'].max()}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{results_df['roc_auc'].mean():.4f}"
    )

    print(
        f"ROC-AUC SD: "
        f"{results_df['roc_auc'].std(ddof=0):.4f}"
    )

    print(
        f"Mean PR-AUC: "
        f"{results_df['pr_auc'].mean():.4f}"
    )

    print(
        f"PR-AUC SD: "
        f"{results_df['pr_auc'].std(ddof=0):.4f}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("PHASE 11I CHECKPOINT 4 COMPLETE")
    print("=" * 70)
    print(f"Folds evaluated: {len(results_df)}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("Validation/test data were not used.")


if __name__ == "__main__":
    main()