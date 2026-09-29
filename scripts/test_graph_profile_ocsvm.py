from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.one_class_svm import GenuineOneClassSVM


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_graph_features.csv"
)

EXPECTED_FEATURE_COUNT = 45


def print_scores(dataframe, scores, title):
    print(f"\n{title}")
    print("-" * 75)

    for (_, row), score in zip(dataframe.iterrows(), scores):
        print(
            f"{row['video_name']:<18} "
            f"{row['label']:<5} "
            f"group={int(row['group_id']):<2} "
            f"score={score: .6f}"
        )


def main():
    print("\nDeepGraph-Phys — Graph-Profile One-Class SVM")
    print("=" * 75)

    # 1. Load graph-profile dataset.
    dataframe = pd.read_csv(DATASET_PATH)

    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith(
            ("motion_gsp_", "physiology_gsp_")
        )
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} graph features, "
            f"found {len(feature_columns)}."
        )

    # 2. Select the predefined splits.
    train = dataframe[dataframe["split"] == "train"].copy()
    validation = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    # Deliberately do not select or inspect test rows here.
    train_real = train[train["label"] == "real"].copy()

    if train_real.empty:
        raise ValueError("No genuine training samples found.")

    if validation["label"].nunique() != 2:
        raise ValueError(
            "Validation must contain both real and fake samples."
        )

    # 3. Extract feature matrices.
    X_train_real = train_real[feature_columns].to_numpy(
        dtype=np.float64
    )
    X_validation = validation[feature_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(X_train_real).all():
        raise ValueError("Genuine training features contain invalid values.")

    if not np.isfinite(X_validation).all():
        raise ValueError("Validation features contain invalid values.")

    # 4. Fit preprocessing ONLY on genuine training videos.
    scaler = StandardScaler()
    X_train_real_scaled = scaler.fit_transform(X_train_real)
    X_validation_scaled = scaler.transform(X_validation)

    # 5. Fit the same One-Class SVM configuration as the 60-D baseline.
    model = GenuineOneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )

    model.fit(X_train_real_scaled)

    # 6. Score genuine training and validation samples.
    # Higher score means more anomalous.
    train_scores = model.anomaly_score(X_train_real_scaled)
    validation_scores = model.anomaly_score(X_validation_scaled)

    validation_auc = roc_auc_score(
        (validation["label"] == "fake").astype(int),
        validation_scores,
    )

    # 7. Print results.
    print("\nDataset and fitting summary:")
    print("Graph feature count:", len(feature_columns))
    print("Genuine training videos:", len(train_real))
    print("Validation videos:", len(validation))
    print("Scaler fitted on genuine training videos only: True")
    print("OCSVM fitted on genuine training videos only: True")
    print("Model: RBF One-Class SVM, gamma='scale', nu=0.10")

    print_scores(
        train_real,
        train_scores,
        "GENUINE TRAINING SCORES",
    )

    print_scores(
        validation,
        validation_scores,
        "VALIDATION SCORES",
    )

    print("\nValidation ROC-AUC:", f"{validation_auc:.6f}")

    y_validation = (
        validation["label"] == "fake"
    ).to_numpy()

    real_scores = validation_scores[~y_validation]
    fake_scores = validation_scores[y_validation]

    print("\nValidation score summary:")
    print("Real mean:", f"{np.mean(real_scores):.6f}")
    print("Fake mean:", f"{np.mean(fake_scores):.6f}")
    print("Real median:", f"{np.median(real_scores):.6f}")
    print("Fake median:", f"{np.median(fake_scores):.6f}")

    print("\nIntegrity checks:")
    print("Model fitted:", model.is_fitted)
    print("Training scores finite:", np.isfinite(train_scores).all())
    print("Validation scores finite:", np.isfinite(validation_scores).all())

    print("\nProtocol:")
    print("- Model and scaler fitted only on genuine training videos.")
    print("- Validation labels used only for ROC-AUC comparison.")
    print("- No classification threshold selected.")
    print("- Test rows were not loaded or evaluated.")


if __name__ == "__main__":
    main()