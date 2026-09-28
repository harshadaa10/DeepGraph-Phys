from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.models.anomaly_data import (
    prepare_anomaly_data,
)

from src.models.isolation_forest import (
    GenuineIsolationForest,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_features.csv"
)


def print_scores(
    dataframe,
    scores,
    title,
):
    print(
        f"\n{title}"
    )

    print("-" * 75)

    for (_, row), score in zip(
        dataframe.iterrows(),
        scores,
    ):

        print(
            f"{row['video_name']:<18} "
            f"{row['label']:<5} "
            f"group={int(row['group_id']):<2} "
            f"score={score: .6f}"
        )


def main():

    print(
        "\nDeepGraph-Phys — "
        "Genuine Isolation Forest Test"
    )

    print("=" * 75)

    data = prepare_anomaly_data(
        DATASET_PATH
    )

    # ------------------------------------------
    # Train ONLY on genuine training videos
    # ------------------------------------------

    model = GenuineIsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    model.fit(
        data[
            "X_train_real_scaled"
        ]
    )

    # ------------------------------------------
    # Obtain continuous anomaly scores
    # ------------------------------------------

    train_scores = (
        model.anomaly_score(
            data[
                "X_train_real_scaled"
            ]
        )
    )

    validation_scores = (
        model.anomaly_score(
            data[
                "X_validation_scaled"
            ]
        )
    )

    # IMPORTANT:
    # We compute test scores here only as an
    # implementation sanity check.
    #
    # Do NOT inspect/use them for model or
    # threshold selection in this phase.
    test_scores = (
        model.anomaly_score(
            data[
                "X_test_scaled"
            ]
        )
    )

    # ------------------------------------------
    # Validation ROC-AUC
    #
    # y:
    # 0 = real
    # 1 = fake
    #
    # score:
    # higher = more anomalous
    # ------------------------------------------

    validation_auc = roc_auc_score(
        data[
            "y_validation"
        ],
        validation_scores,
    )

    # ------------------------------------------
    # Print train + validation only
    # ------------------------------------------

    print_scores(
        data[
            "train_real_dataframe"
        ],
        train_scores,
        "GENUINE TRAINING SCORES",
    )

    print_scores(
        data[
            "validation_dataframe"
        ],
        validation_scores,
        "VALIDATION SCORES",
    )

    print(
        "\nValidation ROC-AUC:",
        f"{validation_auc:.6f}",
    )

    # ------------------------------------------
    # Summary by validation class
    # ------------------------------------------

    y_validation = data[
        "y_validation"
    ]

    real_scores = (
        validation_scores[
            y_validation == 0
        ]
    )

    fake_scores = (
        validation_scores[
            y_validation == 1
        ]
    )

    print(
        "\nValidation score summary:"
    )

    print(
        "Real mean:",
        f"{np.mean(real_scores):.6f}",
    )

    print(
        "Fake mean:",
        f"{np.mean(fake_scores):.6f}",
    )

    print(
        "Real median:",
        f"{np.median(real_scores):.6f}",
    )

    print(
        "Fake median:",
        f"{np.median(fake_scores):.6f}",
    )

    # ------------------------------------------
    # Integrity checks
    # ------------------------------------------

    print(
        "\nValidation checks:"
    )

    print(
        "Model fitted:",
        model.is_fitted,
    )

    print(
        "Train scores finite:",
        np.isfinite(
            train_scores
        ).all(),
    )

    print(
        "Validation scores finite:",
        np.isfinite(
            validation_scores
        ).all(),
    )

    print(
        "Test scores finite:",
        np.isfinite(
            test_scores
        ).all(),
    )

    print(
        "Train score count:",
        len(train_scores),
    )

    print(
        "Validation score count:",
        len(validation_scores),
    )

    print(
        "Test score count:",
        len(test_scores),
    )

    print(
        "\nNOTE:"
    )

    print(
        "Higher score means more anomalous."
    )

    print(
        "Test scores were intentionally "
        "not displayed."
    )

    print(
        "Do not select a threshold from "
        "the test set."
    )


if __name__ == "__main__":
    main()