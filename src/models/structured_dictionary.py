import numpy as np

from src.models.sparse_dictionary import (
    GenuineSparseDictionary,
)


class StructuredGenuineDictionary:
    """
    Structured genuine-only anomaly detector.

    DeepGraph-Phys features are divided into:

        Motion/GSP features      : columns 0-29
        Physiology/GSP features  : columns 30-59

    A separate sparse dictionary is learned
    for each modality using genuine videos only.

    Final anomaly score:

        motion_weight * motion_error
        +
        physiology_weight * physiology_error

    Higher score = more anomalous.
    """

    def __init__(
        self,
        motion_components=6,
        physiology_components=6,
        alpha=1.0,
        transform_alpha=1.0,
        motion_weight=0.5,
        physiology_weight=0.5,
        max_iter=1000,
        random_state=42,
    ):

        if motion_weight < 0:
            raise ValueError(
                "motion_weight cannot be negative."
            )

        if physiology_weight < 0:
            raise ValueError(
                "physiology_weight cannot be negative."
            )

        total_weight = (
            motion_weight
            + physiology_weight
        )

        if not np.isclose(
            total_weight,
            1.0,
        ):
            raise ValueError(
                "Motion and physiology weights "
                "must sum to 1."
            )

        self.motion_weight = motion_weight
        self.physiology_weight = (
            physiology_weight
        )

        self.motion_model = (
            GenuineSparseDictionary(
                n_components=motion_components,
                alpha=alpha,
                transform_alpha=transform_alpha,
                max_iter=max_iter,
                random_state=random_state,
            )
        )

        self.physiology_model = (
            GenuineSparseDictionary(
                n_components=physiology_components,
                alpha=alpha,
                transform_alpha=transform_alpha,
                max_iter=max_iter,
                random_state=random_state,
            )
        )

        self.is_fitted = False

    def _validate_matrix(
        self,
        X,
        name,
    ):

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D matrix."
            )

        if X.shape[0] == 0:
            raise ValueError(
                f"{name} contains no samples."
            )

        if X.shape[1] != 60:
            raise ValueError(
                f"{name} must contain exactly "
                "60 DeepGraph-Phys features."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"{name} contains NaN or "
                "infinite values."
            )

        return X

    def _split_modalities(
        self,
        X,
    ):
        """
        Split the fixed DeepGraph-Phys feature
        vector into motion and physiology groups.
        """

        X = self._validate_matrix(
            X,
            "X",
        )

        motion = X[:, :30]

        physiology = X[:, 30:]

        return (
            motion,
            physiology,
        )

    def fit(
        self,
        X_train_real,
    ):
        """
        Fit both modality dictionaries using
        genuine training samples only.
        """

        X_train_real = (
            self._validate_matrix(
                X_train_real,
                "X_train_real",
            )
        )

        (
            motion,
            physiology,
        ) = self._split_modalities(
            X_train_real
        )

        self.motion_model.fit(
            motion
        )

        self.physiology_model.fit(
            physiology
        )

        self.is_fitted = True

        return self

    def modality_scores(
        self,
        X,
    ):
        """
        Return separate reconstruction errors
        for motion and physiology.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "scoring."
            )

        X = self._validate_matrix(
            X,
            "X",
        )

        (
            motion,
            physiology,
        ) = self._split_modalities(
            X
        )

        motion_scores = (
            self.motion_model.anomaly_score(
                motion
            )
        )

        physiology_scores = (
            self.physiology_model.anomaly_score(
                physiology
            )
        )

        return (
            motion_scores,
            physiology_scores,
        )

    def anomaly_score(
        self,
        X,
    ):
        """
        Return combined structured anomaly score.
        """

        (
            motion_scores,
            physiology_scores,
        ) = self.modality_scores(
            X
        )

        combined_scores = (
            self.motion_weight
            * motion_scores
            +
            self.physiology_weight
            * physiology_scores
        )

        return np.asarray(
            combined_scores,
            dtype=np.float64,
        )