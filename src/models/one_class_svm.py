import numpy as np
from sklearn.svm import OneClassSVM


class GenuineOneClassSVM:
    """
    Genuine-only One-Class SVM baseline.

    Higher DeepGraph-Phys anomaly score means
    more anomalous / more suspicious.
    """

    def __init__(
        self,
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    ):
        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu,
        )

        self.is_fitted = False

    def fit(
        self,
        X_train_real,
    ):
        X_train_real = np.asarray(
            X_train_real,
            dtype=np.float64,
        )

        if X_train_real.ndim != 2:
            raise ValueError(
                "Training data must be 2D."
            )

        if X_train_real.shape[0] == 0:
            raise ValueError(
                "No training samples provided."
            )

        if not np.isfinite(
            X_train_real
        ).all():
            raise ValueError(
                "Training data contains "
                "invalid values."
            )

        self.model.fit(
            X_train_real
        )

        self.is_fitted = True

        return self

    def anomaly_score(
        self,
        X,
    ):
        """
        sklearn decision_function:
            larger = more normal

        DeepGraph-Phys convention:
            larger = more anomalous
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "scoring."
            )

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "Input data must be 2D."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Input contains invalid values."
            )

        scores = (
            -self.model.decision_function(X)
        )

        return np.asarray(
            scores,
            dtype=np.float64,
        )