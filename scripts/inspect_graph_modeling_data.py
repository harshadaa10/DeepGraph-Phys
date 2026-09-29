from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ffpp_graph_features.csv"
)


def main():

    print(
        "\nDeepGraph-Phys — "
        "Graph Modeling Data Inspection"
    )

    print("=" * 70)

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith(
            (
                "motion_gsp_",
                "physiology_gsp_",
            )
        )
    ]

    motion_columns = [
        column
        for column in feature_columns
        if column.startswith(
            "motion_gsp_"
        )
    ]

    physiology_columns = [
        column
        for column in feature_columns
        if column.startswith(
            "physiology_gsp_"
        )
    ]

    train = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    test = dataframe[
        dataframe["split"] == "test"
    ].copy()

    genuine_train = train[
        train["label"] == "real"
    ].copy()

    fake_train = train[
        train["label"] == "fake"
    ].copy()

    print(
        "\nDataset:"
    )

    print(
        "Rows:",
        len(dataframe),
    )

    print(
        "Graph features:",
        len(feature_columns),
    )

    print(
        "Motion features:",
        len(motion_columns),
    )

    print(
        "Physiology features:",
        len(physiology_columns),
    )

    print(
        "\nSplit sizes:"
    )

    print(
        "Train:",
        len(train),
    )

    print(
        "Validation:",
        len(validation),
    )

    print(
        "Test:",
        len(test),
    )

    print(
        "\nTraining composition:"
    )

    print(
        "Real:",
        len(genuine_train),
    )

    print(
        "Fake:",
        len(fake_train),
    )

    print(
        "\nModel fitting set:"
    )

    print(
        "Genuine training samples:",
        len(genuine_train),
    )

    X_train = genuine_train[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    print(
        "X_train shape:",
        X_train.shape,
    )

    # ==========================================
    # Feature variability
    # ==========================================

    standard_deviations = np.std(
        X_train,
        axis=0,
    )

    zero_variance = (
        standard_deviations
        <= 1e-12
    )

    near_zero_variance = (
        (
            standard_deviations
            > 1e-12
        )
        & (
            standard_deviations
            <= 1e-6
        )
    )

    print(
        "\nFeature variability:"
    )

    print(
        "Zero-variance features:",
        int(
            np.sum(
                zero_variance
            )
        ),
    )

    print(
        "Near-zero-variance features:",
        int(
            np.sum(
                near_zero_variance
            )
        ),
    )

    if np.any(
        zero_variance
    ):

        print(
            "\nZero-variance names:"
        )

        for name in np.asarray(
            feature_columns
        )[
            zero_variance
        ]:
            print(
                " -",
                name,
            )

    if np.any(
        near_zero_variance
    ):

        print(
            "\nNear-zero-variance names:"
        )

        for name in np.asarray(
            feature_columns
        )[
            near_zero_variance
        ]:
            print(
                " -",
                name,
            )

    # ==========================================
    # Matrix rank
    # ==========================================

    centered = (
        X_train
        - np.mean(
            X_train,
            axis=0,
            keepdims=True,
        )
    )

    rank = np.linalg.matrix_rank(
        centered
    )

    print(
        "\nDimensionality:"
    )

    print(
        "Feature dimension:",
        X_train.shape[1],
    )

    print(
        "Genuine training samples:",
        X_train.shape[0],
    )

    print(
        "Centered matrix rank:",
        rank,
    )

    print(
        "Maximum possible centered rank:",
        min(
            X_train.shape[0] - 1,
            X_train.shape[1],
        ),
    )

    # ==========================================
    # Feature ranges
    # ==========================================

    print(
        "\nGenuine-train feature range:"
    )

    print(
        "Minimum:",
        float(
            np.min(
                X_train
            )
        ),
    )

    print(
        "Maximum:",
        float(
            np.max(
                X_train
            )
        ),
    )

    print(
        "All finite:",
        bool(
            np.isfinite(
                X_train
            ).all()
        ),
    )

    # ==========================================
    # Validation availability
    # ==========================================

    print(
        "\nValidation composition:"
    )

    print(
        validation[
            "label"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTest rows reserved and untouched:",
        len(test),
    )

    print(
        "\nInspection completed."
    )


if __name__ == "__main__":
    main()