import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import sparse_encode


class GenuineSparseDictionary:
    """
    Genuine-only sparse dictionary anomaly detector.

    The dictionary is learned exclusively from
    genuine DeepGraph-Phys feature vectors.

    Anomaly score:
        mean squared reconstruction error

    Higher score = more anomalous / suspicious.
    """

    def __init__(
        self,
        n_components=6,
        alpha=1.0,
        transform_alpha=1.0,
        max_iter=1000,
        random_state=42,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.transform_alpha = transform_alpha
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = DictionaryLearning(
            n_components=n_components,
            alpha=alpha,
            max_iter=max_iter,
            fit_algorithm="cd",
            transform_algorithm="lasso_cd",
            transform_alpha=transform_alpha,
            random_state=random_state,
        )

        self.is_fitted = False
        self.dictionary_ = None

    def _validate_matrix(
        self,
        X,
        name,
    ):
        """
        Validate an input feature matrix.
        """

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

        if not np.isfinite(X).all():
            raise ValueError(
                f"{name} contains NaN or "
                "infinite values."
            )

        return X

    def fit(
        self,
        X_train_real,
    ):
        """
        Learn a sparse dictionary using ONLY
        genuine training samples.
        """

        X_train_real = self._validate_matrix(
            X_train_real,
            "X_train_real",
        )

        if self.n_components > X_train_real.shape[0]:
            raise ValueError(
                "Dictionary components cannot "
                "exceed the number of genuine "
                "training samples in this "
                "development implementation."
            )

        self.model.fit(
            X_train_real
        )

        self.dictionary_ = np.asarray(
            self.model.components_,
            dtype=np.float64,
        )

        if not np.isfinite(
            self.dictionary_
        ).all():
            raise ValueError(
                "Learned dictionary contains "
                "invalid values."
            )

        self.is_fitted = True

        return self

    def sparse_codes(
        self,
        X,
    ):
        """
        Compute sparse coefficients for samples.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "encoding samples."
            )

        X = self._validate_matrix(
            X,
            "X",
        )

        codes = sparse_encode(
            X,
            self.dictionary_,
            algorithm="lasso_cd",
            alpha=self.transform_alpha,
        )

        return np.asarray(
            codes,
            dtype=np.float64,
        )

    def reconstruct(
        self,
        X,
    ):
        """
        Reconstruct feature vectors using
        the genuine dictionary.
        """

        X = self._validate_matrix(
            X,
            "X",
        )

        codes = self.sparse_codes(
            X
        )

        reconstructed = (
            codes @ self.dictionary_
        )

        return np.asarray(
            reconstructed,
            dtype=np.float64,
        )

    def anomaly_score(
        self,
        X,
    ):
        """
        Calculate mean squared reconstruction
        error for each sample.

        Higher reconstruction error means the
        sample is less compatible with the
        learned genuine dictionary.
        """

        X = self._validate_matrix(
            X,
            "X",
        )

        reconstructed = self.reconstruct(
            X
        )

        errors = np.mean(
            np.square(
                X - reconstructed
            ),
            axis=1,
        )

        return np.asarray(
            errors,
            dtype=np.float64,
        )