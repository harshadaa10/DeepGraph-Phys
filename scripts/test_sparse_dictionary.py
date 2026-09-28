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

from src.models.sparse_dictionary import (
    GenuineSparseDictionary,
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
        "Genuine Sparse Dictionary Test"
    )

    print("=" * 75)

    # ==================================================
    # 1. Prepare leakage-safe data
    # ==================================================

    data = prepare_anomaly_data(
        DATASET_PATH
    )

    # ==================================================
    # 2. Create dictionary model
    #
    # IMPORTANT:
    # These are fixed development parameters.
    # We are NOT tuning them using fake validation
    # videos at this stage.
    # ==================================================

    model = GenuineSparseDictionary(
        n_components=6,
        alpha=1.0,
        transform_alpha=1.0,
        max_iter=1000,
        random_state=42,
    )

    # ==================================================
    # 3. Train ONLY on genuine training videos
    # ==================================================

    model.fit(
        data[
            "X_train_real_scaled"
        ]
    )

    # ==================================================
    # 4. Inspect learned dictionary
    # ==================================================

    print(
        "\nDictionary shape:",
        model.dictionary_.shape,
    )

    print(
        "Expected dictionary shape:",
        "(6, 60)",
    )

    # ==================================================
    # 5. Calculate anomaly scores
    #
    # Higher reconstruction error = more anomalous
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

    # Test is computed ONLY as an implementation
    # sanity check.
    #
    # Do not print or use the test scores for
    # model selection.
    test_scores = (
        model.anomaly_score(
            data[
                "X_test_scaled"
            ]
        )
    )

    # ==================================================
    # 6. Validation ROC-AUC
    #
    # 0 = real
    # 1 = fake
    #
    # Higher score = more anomalous
    # ==================================================

    validation_auc = roc_auc_score(
        data[
            "y_validation"
        ],
        validation_scores,
    )

    # ==================================================
    # 7. Display training scores
    # ==================================================

    print_scores(
        data[
            "train_real_dataframe"
        ],
        train_scores,
        "GENUINE TRAINING SCORES",
    )

    # ==================================================
    # 8. Display validation scores
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
    # 9. Validation class summary
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
    # 10. Sparse-code statistics
    # ==================================================

    train_codes = model.sparse_codes(
        data[
            "X_train_real_scaled"
        ]
    )

    validation_codes = model.sparse_codes(
        data[
            "X_validation_scaled"
        ]
    )

    print(
        "\nSparse representation:"
    )

    print(
        "Training code shape:",
        train_codes.shape,
    )

    print(
        "Validation code shape:",
        validation_codes.shape,
    )

    train_nonzero = np.sum(
        np.abs(train_codes) > 1e-8,
        axis=1,
    )

    validation_nonzero = np.sum(
        np.abs(validation_codes) > 1e-8,
        axis=1,
    )

    print(
        "Mean active atoms — train:",
        f"{np.mean(train_nonzero):.3f}",
    )

    print(
        "Mean active atoms — validation:",
        f"{np.mean(validation_nonzero):.3f}",
    )

    # ==================================================
    # 11. Integrity checks
    # ==================================================

    print(
        "\nValidation checks:"
    )

    print(
        "Model fitted:",
        model.is_fitted,
    )

    print(
        "Dictionary finite:",
        np.isfinite(
            model.dictionary_
        ).all(),
    )

    print(
        "Dictionary shape correct:",
        model.dictionary_.shape
        == (6, 60),
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
        "All train scores non-negative:",
        np.all(
            train_scores >= 0
        ),
    )

    print(
        "All validation scores non-negative:",
        np.all(
            validation_scores >= 0
        ),
    )

    # ==================================================
    # 12. Reminder
    # ==================================================

    print(
        "\nNOTE:"
    )

    print(
        "Higher reconstruction error means "
        "more anomalous."
    )

    print(
        "Dictionary was learned only from "
        "genuine training videos."
    )

    print(
        "Validation labels were used only "
        "for evaluation, not fitting."
    )

    print(
        "Test scores were intentionally "
        "not displayed."
    )

    print(
        "Do not choose parameters or a "
        "threshold from the test set."
    )


if __name__ == "__main__":
    main()