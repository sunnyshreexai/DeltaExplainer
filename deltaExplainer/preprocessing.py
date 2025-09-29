"""Data preprocessing utilities for DeltaExplainer."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class DataPreprocessor:
    """Preprocessing utilities for tabular data."""

    def __init__(self):
        """Initialize preprocessor."""
        self.feature_types = {}
        self.scalers = {}
        self.encoders = {}
        self.mad_values = {}
        self.feature_names = []
        self.is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_types: Optional[Dict[int, str]] = None,
    ) -> "DataPreprocessor":
        """Fit preprocessor on training data.

        Args:
            X: Training data
            feature_types: Dict mapping feature index to type ('continuous' or 'categorical')
                          If None, will be inferred from data

        Returns:
            Self
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.feature_names = list(X.columns)
        n_features = len(self.feature_names)

        # Infer feature types if not provided
        if feature_types is None:
            self.feature_types = self._infer_feature_types(X)
        else:
            self.feature_types = feature_types

        # Fit scalers and encoders for each feature
        for idx in range(n_features):
            col = X.iloc[:, idx]

            if self.feature_types.get(idx, "continuous") == "continuous":
                # Fit scaler for continuous features
                scaler = MinMaxScaler()
                scaler.fit(col.values.reshape(-1, 1))
                self.scalers[idx] = scaler

                # Compute MAD (Median Absolute Deviation)
                median = np.median(col)
                self.mad_values[idx] = np.median(np.abs(col - median))
                # Handle zero MAD
                if self.mad_values[idx] == 0:
                    self.mad_values[idx] = np.std(col)
                    if self.mad_values[idx] == 0:
                        self.mad_values[idx] = 1.0

            else:  # categorical
                # Fit encoder for categorical features
                encoder = LabelEncoder()
                encoder.fit(col)
                self.encoders[idx] = encoder
                # MAD is 1 for categorical (as per paper)
                self.mad_values[idx] = 1.0

        self.is_fitted = True
        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform data to normalized form.

        Args:
            X: Data to transform

        Returns:
            Tuple of (transformed_data, continuous_mask, categorical_mask)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        n_samples = len(X)
        n_features = len(self.feature_names)

        # Initialize output arrays
        X_transformed = np.zeros((n_samples, n_features))
        continuous_mask = np.zeros(n_features, dtype=bool)
        categorical_mask = np.zeros(n_features, dtype=bool)

        # Transform each feature
        for idx in range(n_features):
            col = X.iloc[:, idx]

            if self.feature_types.get(idx, "continuous") == "continuous":
                # Scale continuous features to [0, 1]
                X_transformed[:, idx] = self.scalers[idx].transform(
                    col.values.reshape(-1, 1)
                ).flatten()
                continuous_mask[idx] = True
            else:  # categorical
                # Encode categorical features
                X_transformed[:, idx] = self.encoders[idx].transform(col)
                # Normalize to [0, 1] if multiple categories
                n_classes = len(self.encoders[idx].classes_)
                if n_classes > 1:
                    X_transformed[:, idx] /= (n_classes - 1)
                categorical_mask[idx] = True

        return X_transformed, continuous_mask, categorical_mask

    def inverse_transform(
        self, X_transformed: np.ndarray
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Inverse transform data back to original scale.

        Args:
            X_transformed: Transformed data

        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")

        n_samples = X_transformed.shape[0]
        n_features = len(self.feature_names)

        # Initialize output
        X_original = np.zeros((n_samples, n_features))

        # Inverse transform each feature
        for idx in range(n_features):
            col = X_transformed[:, idx]

            if self.feature_types.get(idx, "continuous") == "continuous":
                # Inverse scale continuous features
                X_original[:, idx] = self.scalers[idx].inverse_transform(
                    col.reshape(-1, 1)
                ).flatten()
            else:  # categorical
                # Decode categorical features
                n_classes = len(self.encoders[idx].classes_)
                if n_classes > 1:
                    # Denormalize
                    col = col * (n_classes - 1)
                # Round to nearest integer for categorical
                col = np.round(col).astype(int)
                # Clip to valid range
                col = np.clip(col, 0, n_classes - 1)
                X_original[:, idx] = self.encoders[idx].inverse_transform(col)

        return pd.DataFrame(X_original, columns=self.feature_names)

    def get_mad_array(self) -> np.ndarray:
        """Get MAD values as array.

        Returns:
            Array of MAD values for each feature
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted")

        n_features = len(self.feature_names)
        mad_array = np.zeros(n_features)
        for idx in range(n_features):
            mad_array[idx] = self.mad_values.get(idx, 1.0)

        return mad_array

    def _infer_feature_types(self, X: pd.DataFrame) -> Dict[int, str]:
        """Infer feature types from data.

        Args:
            X: Input data

        Returns:
            Dict mapping feature index to type
        """
        feature_types = {}

        for idx, col in enumerate(X.columns):
            data = X[col]

            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(data):
                # Check number of unique values
                n_unique = len(data.unique())
                n_samples = len(data)

                # If few unique values relative to samples, treat as categorical
                if n_unique < min(10, n_samples * 0.05):
                    feature_types[idx] = "categorical"
                else:
                    feature_types[idx] = "continuous"
            else:
                # Non-numeric columns are categorical
                feature_types[idx] = "categorical"

        return feature_types