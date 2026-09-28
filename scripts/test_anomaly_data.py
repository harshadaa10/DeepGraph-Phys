from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.models.anomaly_data import (
    prepare_anomaly_data,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_features.csv"
)


def main():

    print(
        "\nDeepGraph-Phys — "
        "Anomaly Data Preparation Test"
    )

    print("-" * 60)

    data = prepare_anomaly_data(
        DATASET_PATH
    )

    train_df = (
        data[
            "train_real_dataframe"
        ]
    )

    validation_df = (
        data[
            "validation_dataframe"
        ]
    )

    test_df = (
        data[
            "test_dataframe"
        ]
    )

    print(
        "\nFeature count:",
        len(
            data[
                "feature_columns"
            ]
        ),
    )

    print(
        "\nTraining:"
    )

    print(
        "Real training videos:",
        len(train_df),
    )

    print(
        "Training labels:"
    )

    print(
        train_df[
            "label"
        ].value_counts()
    )

    print(
        "Training groups:",
        sorted(
            train_df[
                "group_id"
            ].unique()
        ),
    )

    print(
        "\nValidation:"
    )

    print(
        validation_df[
            "label"
        ].value_counts()
    )

    print(
        "Validation groups:",
        sorted(
            validation_df[
                "group_id"
            ].unique()
        ),
    )

    print(
        "\nTest:"
    )

    print(
        test_df[
            "label"
        ].value_counts()
    )

    print(
        "Test groups:",
        sorted(
            test_df[
                "group_id"
            ].unique()
        ),
    )

    print(
        "\nMatrix shapes:"
    )

    print(
        "X_train_real:",
        data[
            "X_train_real"
        ].shape,
    )

    print(
        "X_validation:",
        data[
            "X_validation"
        ].shape,
    )

    print(
        "X_test:",
        data[
            "X_test"
        ].shape,
    )

    print(
        "\nScaled matrix shapes:"
    )

    print(
        "Train:",
        data[
            "X_train_real_scaled"
        ].shape,
    )

    print(
        "Validation:",
        data[
            "X_validation_scaled"
        ].shape,
    )

    print(
        "Test:",
        data[
            "X_test_scaled"
        ].shape,
    )

    print(
        "\nValidation labels:",
        data[
            "y_validation"
        ],
    )

    print(
        "Test labels:",
        data[
            "y_test"
        ],
    )

    # ------------------------------------------
    # Leakage / integrity checks
    # ------------------------------------------

    train_groups = set(
        train_df[
            "group_id"
        ]
    )

    validation_groups = set(
        validation_df[
            "group_id"
        ]
    )

    test_groups = set(
        test_df[
            "group_id"
        ]
    )

    group_leakage = (
        bool(
            train_groups
            & validation_groups
        )
        or bool(
            train_groups
            & test_groups
        )
        or bool(
            validation_groups
            & test_groups
        )
    )

    print(
        "\nValidation checks:"
    )

    print(
        "Exactly 60 features:",
        len(
            data[
                "feature_columns"
            ]
        ) == 60,
    )

    print(
        "Exactly 12 genuine "
        "training samples:",
        (
            data[
                "X_train_real"
            ].shape
            == (12, 60)
        ),
    )

    print(
        "Validation shape correct:",
        (
            data[
                "X_validation"
            ].shape
            == (8, 60)
        ),
    )

    print(
        "Test shape correct:",
        (
            data[
                "X_test"
            ].shape
            == (8, 60)
        ),
    )

    print(
        "Training contains only real:",
        set(
            train_df[
                "label"
            ]
        ) == {"real"},
    )

    print(
        "No group leakage:",
        not group_leakage,
    )

    print(
        "Scaled train finite:",
        np.isfinite(
            data[
                "X_train_real_scaled"
            ]
        ).all(),
    )

    print(
        "Scaled validation finite:",
        np.isfinite(
            data[
                "X_validation_scaled"
            ]
        ).all(),
    )

    print(
        "Scaled test finite:",
        np.isfinite(
            data[
                "X_test_scaled"
            ]
        ).all(),
    )

    # StandardScaler fitted on train should
    # produce approximately zero mean.
    train_means = np.mean(
        data[
            "X_train_real_scaled"
        ],
        axis=0,
    )

    print(
        "Scaled training mean "
        "approximately zero:",
        np.allclose(
            train_means,
            0.0,
            atol=1e-7,
        ),
    )


if __name__ == "__main__":
    main()