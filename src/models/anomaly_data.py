from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = [
    "video_name",
    "video_path",
    "label",
    "dataset",
    "manipulation",
    "source_id",
    "target_id",
    "group_id",
    "split",
]


QC_COLUMNS = [
    "fps",
    "frames_processed",
    "rgb_detected_frames",
    "face_detection_rate",
    "rppg_samples",
    "rppg_segment_count",
    "shortest_rppg_segment",
    "longest_rppg_segment",
    "interpolated_frames",
    "interpolation_rate",
    "motion_samples",
    "graph_samples",
    "consecutive_graph_transitions",
    "graph_gap_transitions",
    "largest_graph_frame_difference",
    "synchronized_interpolated_frames",
    "synchronized_interpolation_rate",
    "feature_count",
]


EXPECTED_FEATURE_COUNT = 60


def load_feature_dataset(dataset_path):
    """
    Load the DeepGraph-Phys feature dataset.
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {dataset_path}"
        )

    dataframe = pd.read_csv(dataset_path)

    if dataframe.empty:
        raise ValueError(
            "Feature dataset is empty."
        )

    return dataframe


def get_feature_columns(dataframe):
    """
    Return only the ML feature columns.

    Metadata and quality-control information
    must never enter the anomaly model.
    """

    excluded_columns = (
        METADATA_COLUMNS
        + QC_COLUMNS
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_FEATURE_COUNT} ML features, "
            f"found {len(feature_columns)}."
        )

    return feature_columns


def validate_feature_matrix(
    matrix,
    name,
):
    """
    Validate an ML feature matrix.
    """

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D matrix."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            f"{name} contains no samples."
        )

    if matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{name} must contain "
            f"{EXPECTED_FEATURE_COUNT} features. "
            f"Found {matrix.shape[1]}."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{name} contains NaN or "
            "infinite values."
        )

    return matrix


def prepare_anomaly_data(
    dataset_path,
):
    """
    Prepare leakage-safe data for genuine-only
    anomaly detection.

    Training:
        REAL training videos only.

    Validation:
        All validation videos.

    Test:
        All test videos.

    The scaler is fitted ONLY on genuine
    training samples.
    """

    dataframe = load_feature_dataset(
        dataset_path
    )

    feature_columns = get_feature_columns(
        dataframe
    )

    # ------------------------------------------
    # Split the dataset
    # ------------------------------------------

    train_real = dataframe[
        (dataframe["split"] == "train")
        & (dataframe["label"] == "real")
    ].copy()

    validation = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    test = dataframe[
        dataframe["split"] == "test"
    ].copy()

    # ------------------------------------------
    # Validate expected development split
    # ------------------------------------------

    if len(train_real) == 0:
        raise ValueError(
            "No genuine training samples found."
        )

    if len(validation) == 0:
        raise ValueError(
            "No validation samples found."
        )

    if len(test) == 0:
        raise ValueError(
            "No test samples found."
        )

    # ------------------------------------------
    # Raw ML matrices
    # ------------------------------------------

    X_train_real = (
        train_real[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X_validation = (
        validation[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X_test = (
        test[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X_train_real = validate_feature_matrix(
        X_train_real,
        "X_train_real",
    )

    X_validation = validate_feature_matrix(
        X_validation,
        "X_validation",
    )

    X_test = validate_feature_matrix(
        X_test,
        "X_test",
    )

    # ------------------------------------------
    # Labels
    #
    # 0 = genuine / normal
    # 1 = fake / anomaly
    # ------------------------------------------

    label_mapping = {
        "real": 0,
        "fake": 1,
    }

    validation_labels = (
        validation["label"]
        .map(label_mapping)
    )

    test_labels = (
        test["label"]
        .map(label_mapping)
    )

    # Validate BEFORE integer conversion.
    if (
        validation_labels.isna().any()
        or test_labels.isna().any()
    ):
        raise ValueError(
            "Unexpected label found. "
            "Expected only 'real' or 'fake'."
        )

    y_validation = (
        validation_labels
        .to_numpy(
            dtype=np.int64
        )
    )

    y_test = (
        test_labels
        .to_numpy(
            dtype=np.int64
        )
    )
    if (
        pd.isna(y_validation).any()
        or pd.isna(y_test).any()
    ):
        raise ValueError(
            "Unexpected label found."
        )

    # ------------------------------------------
    # Scaling
    #
    # IMPORTANT:
    # Fit only on REAL training videos.
    # ------------------------------------------

    scaler = StandardScaler()

    X_train_real_scaled = (
        scaler.fit_transform(
            X_train_real
        )
    )

    X_validation_scaled = (
        scaler.transform(
            X_validation
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # ------------------------------------------
    # Final finite-value validation
    # ------------------------------------------

    for name, matrix in [
        (
            "X_train_real_scaled",
            X_train_real_scaled,
        ),
        (
            "X_validation_scaled",
            X_validation_scaled,
        ),
        (
            "X_test_scaled",
            X_test_scaled,
        ),
    ]:

        if not np.isfinite(
            matrix
        ).all():
            raise ValueError(
                f"{name} contains invalid values "
                "after scaling."
            )

    return {
        "dataframe":
            dataframe,

        "feature_columns":
            feature_columns,

        "train_real_dataframe":
            train_real,

        "validation_dataframe":
            validation,

        "test_dataframe":
            test,

        "X_train_real":
            X_train_real,

        "X_validation":
            X_validation,

        "X_test":
            X_test,

        "X_train_real_scaled":
            X_train_real_scaled,

        "X_validation_scaled":
            X_validation_scaled,

        "X_test_scaled":
            X_test_scaled,

        "y_validation":
            y_validation,

        "y_test":
            y_test,

        "scaler":
            scaler,
    }