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

from src.models.one_class_svm import (
    GenuineOneClassSVM,
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
    """
    Print anomaly scores together with
    video identity and ground-truth label.
    """

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
        "Genuine One-Class SVM Test"
    )

    print("=" * 75)

    # ==================================================
    # 1. Prepare leakage-safe data
    # ==================================================

    data = prepare_anomaly_data(
        DATASET_PATH
    )

    # ==================================================
    # 2. Create One-Class SVM
    #
    # IMPORTANT:
    # The model will be trained ONLY on genuine
    # training videos.
    # ==================================================

    model = GenuineOneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )

    # ==================================================
    # 3. Fit only on genuine training samples
    # ==================================================

    model.fit(
        data[
            "X_train_real_scaled"
        ]
    )

    # ==================================================
    # 4. Obtain continuous anomaly scores
    #
    # DeepGraph-Phys convention:
    #
    # higher score = more anomalous
    #              = more suspicious
    # ==================================================

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

    # --------------------------------------------------
    # Test scores are calculated only as an
    # implementation sanity check.
    #
    # They are NOT printed and must NOT be used
    # for parameter or threshold selection.
    # --------------------------------------------------

    test_scores = (
        model.anomaly_score(
            data[
                "X_test_scaled"
            ]
        )
    )

    # ==================================================
    # 5. Validation ROC-AUC
    #
    # Labels:
    #     0 = real / genuine
    #     1 = fake / anomaly
    #
    # Scores:
    #     higher = more anomalous
    # ==================================================

    validation_auc = roc_auc_score(
        data[
            "y_validation"
        ],
        validation_scores,
    )

    # ==================================================
    # 6. Print genuine training scores
    # ==================================================

    print_scores(
        data[
            "train_real_dataframe"
        ],
        train_scores,
        "GENUINE TRAINING SCORES",
    )

    # ==================================================
    # 7. Print validation scores
    # ==================================================

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

    # ==================================================
    # 8. Validation score summary by class
    # ==================================================

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

    # ==================================================
    # 9. Integrity checks
    # ==================================================

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
        len(
            train_scores
        ),
    )

    print(
        "Validation score count:",
        len(
            validation_scores
        ),
    )

    print(
        "Test score count:",
        len(
            test_scores
        ),
    )

    # ==================================================
    # 10. Final reminder
    # ==================================================

    print(
        "\nNOTE:"
    )

    print(
        "Higher score means more anomalous."
    )

    print(
        "One-Class SVM was fitted only "
        "on genuine training videos."
    )

    print(
        "Test scores were intentionally "
        "not displayed."
    )

    print(
        "Do not select model parameters or "
        "a threshold from the test set."
    )


if __name__ == "__main__":
    main()