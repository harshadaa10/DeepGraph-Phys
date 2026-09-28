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

from src.models.structured_dictionary import (
    StructuredGenuineDictionary,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_features.csv"
)


def print_scores(
    dataframe,
    total_scores,
    motion_scores,
    physiology_scores,
    title,
):

    print(
        f"\n{title}"
    )

    print("-" * 105)

    print(
        f"{'Video':<18} "
        f"{'Label':<6} "
        f"{'Group':<7} "
        f"{'Motion':>12} "
        f"{'Physiology':>14} "
        f"{'Combined':>12}"
    )

    print("-" * 105)

    for (
        (_, row),
        total,
        motion,
        physiology,
    ) in zip(
        dataframe.iterrows(),
        total_scores,
        motion_scores,
        physiology_scores,
    ):

        print(
            f"{row['video_name']:<18} "
            f"{row['label']:<6} "
            f"{int(row['group_id']):<7} "
            f"{motion:>12.6f} "
            f"{physiology:>14.6f} "
            f"{total:>12.6f}"
        )


def main():

    print(
        "\nDeepGraph-Phys — "
        "Structured Genuine Dictionary Test"
    )

    print("=" * 105)

    # ==================================================
    # 1. Leakage-safe data preparation
    # ==================================================

    data = prepare_anomaly_data(
        DATASET_PATH
    )

    # ==================================================
    # 2. Structured model
    #
    # Equal modality weights are fixed BEFORE
    # inspecting validation performance.
    # ==================================================

    model = StructuredGenuineDictionary(
        motion_components=6,
        physiology_components=6,
        alpha=1.0,
        transform_alpha=1.0,
        motion_weight=0.5,
        physiology_weight=0.5,
        max_iter=1000,
        random_state=42,
    )

    # ==================================================
    # 3. Genuine-only fitting
    # ==================================================

    model.fit(
        data[
            "X_train_real_scaled"
        ]
    )

    # ==================================================
    # 4. Training scores
    # ==================================================

    train_total = (
        model.anomaly_score(
            data[
                "X_train_real_scaled"
            ]
        )
    )

    (
        train_motion,
        train_physiology,
    ) = model.modality_scores(
        data[
            "X_train_real_scaled"
        ]
    )

    # ==================================================
    # 5. Validation scores
    # ==================================================

    validation_total = (
        model.anomaly_score(
            data[
                "X_validation_scaled"
            ]
        )
    )

    (
        validation_motion,
        validation_physiology,
    ) = model.modality_scores(
        data[
            "X_validation_scaled"
        ]
    )

    # ==================================================
    # 6. Test sanity check only
    #
    # Do NOT print test scores.
    # ==================================================

    test_total = (
        model.anomaly_score(
            data[
                "X_test_scaled"
            ]
        )
    )

    (
        test_motion,
        test_physiology,
    ) = model.modality_scores(
        data[
            "X_test_scaled"
        ]
    )

    # ==================================================
    # 7. Print scores
    # ==================================================

    print_scores(
        data[
            "train_real_dataframe"
        ],
        train_total,
        train_motion,
        train_physiology,
        "GENUINE TRAINING SCORES",
    )

    print_scores(
        data[
            "validation_dataframe"
        ],
        validation_total,
        validation_motion,
        validation_physiology,
        "VALIDATION SCORES",
    )

    # ==================================================
    # 8. ROC-AUC
    # ==================================================

    y_validation = data[
        "y_validation"
    ]

    combined_auc = roc_auc_score(
        y_validation,
        validation_total,
    )

    motion_auc = roc_auc_score(
        y_validation,
        validation_motion,
    )

    physiology_auc = roc_auc_score(
        y_validation,
        validation_physiology,
    )

    print(
        "\nValidation ROC-AUC:"
    )

    print(
        "Motion only:",
        f"{motion_auc:.6f}",
    )

    print(
        "Physiology only:",
        f"{physiology_auc:.6f}",
    )

    print(
        "Combined:",
        f"{combined_auc:.6f}",
    )

    # ==================================================
    # 9. Class-level summaries
    # ==================================================

    real_mask = (
        y_validation == 0
    )

    fake_mask = (
        y_validation == 1
    )

    print(
        "\nMean validation anomaly scores:"
    )

    print(
        "REAL motion:",
        f"{np.mean(validation_motion[real_mask]):.6f}",
    )

    print(
        "FAKE motion:",
        f"{np.mean(validation_motion[fake_mask]):.6f}",
    )

    print(
        "REAL physiology:",
        f"{np.mean(validation_physiology[real_mask]):.6f}",
    )

    print(
        "FAKE physiology:",
        f"{np.mean(validation_physiology[fake_mask]):.6f}",
    )

    print(
        "REAL combined:",
        f"{np.mean(validation_total[real_mask]):.6f}",
    )

    print(
        "FAKE combined:",
        f"{np.mean(validation_total[fake_mask]):.6f}",
    )

    # ==================================================
    # 10. Dictionary information
    # ==================================================

    print(
        "\nDictionary shapes:"
    )

    print(
        "Motion:",
        model.motion_model.dictionary_.shape,
    )

    print(
        "Physiology:",
        model.physiology_model.dictionary_.shape,
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
        "Motion dictionary correct:",
        model.motion_model.dictionary_.shape
        == (6, 30),
    )

    print(
        "Physiology dictionary correct:",
        model.physiology_model.dictionary_.shape
        == (6, 30),
    )

    print(
        "Train scores finite:",
        np.isfinite(
            train_total
        ).all(),
    )

    print(
        "Validation scores finite:",
        np.isfinite(
            validation_total
        ).all(),
    )

    print(
        "Test scores finite:",
        np.isfinite(
            test_total
        ).all(),
    )

    print(
        "Test motion scores finite:",
        np.isfinite(
            test_motion
        ).all(),
    )

    print(
        "Test physiology scores finite:",
        np.isfinite(
            test_physiology
        ).all(),
    )

    print(
        "All validation scores non-negative:",
        np.all(
            validation_total >= 0
        ),
    )

    print(
        "Combined score calculation correct:",
        np.allclose(
            validation_total,
            0.5 * validation_motion
            + 0.5 * validation_physiology,
        ),
    )

    print(
        "\nNOTE:"
    )

    print(
        "Higher reconstruction error means "
        "more anomalous."
    )

    print(
        "Motion and physiology dictionaries "
        "were trained only on genuine videos."
    )

    print(
        "The 0.5 / 0.5 modality weights were "
        "fixed without validation tuning."
    )

    print(
        "Test scores were intentionally "
        "not displayed."
    )


if __name__ == "__main__":
    main()