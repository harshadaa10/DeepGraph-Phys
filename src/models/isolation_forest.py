import numpy as np
from sklearn.ensemble import IsolationForest


class GenuineIsolationForest:
    """
    Genuine-only Isolation Forest baseline.

    The model is trained exclusively on
    genuine facial-behaviour feature vectors.

    Public anomaly scores are oriented so:

        higher score = more anomalous
                     = more suspicious
    """

    def __init__(
        self,
        n_estimators=200,
        contamination="auto",
        random_state=42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

        self.is_fitted = False

    def fit(
        self,
        X_train_real,
    ):
        """
        Fit only on genuine training samples.
        """

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
        Return anomaly-oriented scores.

        sklearn decision_function:
            larger = more normal

        DeepGraph-Phys convention:
            larger = more anomalous

        Therefore we negate the sklearn score.
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

        scores = -self.model.decision_function(
            X
        )

        return np.asarray(
            scores,
            dtype=np.float64,
        )