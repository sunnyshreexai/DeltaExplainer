"""Distance metrics and loss functions for counterfactual generation."""

from typing import Optional, Tuple, Callable
import numpy as np


class DistanceMetrics:
    """Distance metrics for counterfactual explanations."""

    @staticmethod
    def continuous_distance(
        original: np.ndarray,
        counterfactual: np.ndarray,
        mad: np.ndarray,
        continuous_mask: np.ndarray,
    ) -> float:
        """Compute distance for continuous features using MAD normalization.

        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            mad: Median Absolute Deviation for each feature
            continuous_mask: Binary mask for continuous features

        Returns:
            Normalized distance for continuous features
        """
        if not np.any(continuous_mask):
            return 0.0

        # Avoid division by zero
        mad_safe = np.where(mad > 0, mad, 1.0)

        # Compute absolute differences normalized by MAD
        diff = np.abs(counterfactual - original) * continuous_mask
        normalized_diff = diff / mad_safe

        # Average over continuous features
        n_continuous = np.sum(continuous_mask)
        return np.sum(normalized_diff) / max(n_continuous, 1)

    @staticmethod
    def categorical_distance(
        original: np.ndarray,
        counterfactual: np.ndarray,
        categorical_mask: np.ndarray,
    ) -> float:
        """Compute distance for categorical features.

        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            categorical_mask: Binary mask for categorical features

        Returns:
            Normalized distance for categorical features
        """
        if not np.any(categorical_mask):
            return 0.0

        # Count differences in categorical features
        diff = (original != counterfactual) * categorical_mask

        # Average over categorical features
        n_categorical = np.sum(categorical_mask)
        return np.sum(diff) / max(n_categorical, 1)

    @staticmethod
    def total_distance(
        original: np.ndarray,
        counterfactual: np.ndarray,
        mad: np.ndarray,
        continuous_mask: np.ndarray,
        categorical_mask: np.ndarray,
        distance_metric: str = "l1",
    ) -> float:
        """Compute total distance between original and counterfactual.

        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            mad: Median Absolute Deviation for each feature
            continuous_mask: Binary mask for continuous features
            categorical_mask: Binary mask for categorical features
            distance_metric: Type of distance metric ('l1' or 'l2')

        Returns:
            Total distance
        """
        cont_dist = DistanceMetrics.continuous_distance(
            original, counterfactual, mad, continuous_mask
        )
        cat_dist = DistanceMetrics.categorical_distance(
            original, counterfactual, categorical_mask
        )

        if distance_metric == "l2":
            return np.sqrt(cont_dist ** 2 + cat_dist ** 2)
        else:  # l1
            return cont_dist + cat_dist


class LossFunction:
    """Loss function for counterfactual optimization."""

    def __init__(
        self,
        model_predict_proba: Callable[[np.ndarray], np.ndarray],
        original_instance: np.ndarray,
        original_class: int,
        mad: np.ndarray,
        continuous_mask: np.ndarray,
        categorical_mask: np.ndarray,
        lambda_param: float = 1.0,
        distance_metric: str = "l1",
    ):
        """Initialize loss function.

        Args:
            model_predict_proba: Model prediction function returning probabilities
            original_instance: Original instance
            original_class: Original predicted class
            mad: Median Absolute Deviation for each feature
            continuous_mask: Binary mask for continuous features
            categorical_mask: Binary mask for categorical features
            lambda_param: Weight for distance term
            distance_metric: Type of distance metric
        """
        self.model_predict_proba = model_predict_proba
        self.original_instance = original_instance
        self.original_class = original_class
        self.mad = mad
        self.continuous_mask = continuous_mask
        self.categorical_mask = categorical_mask
        self.lambda_param = lambda_param
        self.distance_metric = distance_metric

    def compute(
        self, counterfactual: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute loss and gradient.

        Args:
            counterfactual: Current counterfactual candidate

        Returns:
            Tuple of (loss, gradient)
        """
        # Reshape for model prediction if needed
        cf_reshaped = counterfactual.reshape(1, -1)

        # Get model predictions
        proba = self.model_predict_proba(cf_reshaped)[0]

        # Classification loss (negative log probability of NOT being original class)
        # We want to minimize the probability of original class
        y_loss = proba[self.original_class]

        # Distance loss
        dist_loss = DistanceMetrics.total_distance(
            self.original_instance,
            counterfactual,
            self.mad,
            self.continuous_mask,
            self.categorical_mask,
            self.distance_metric,
        )

        # Total loss
        total_loss = y_loss + self.lambda_param * dist_loss

        # Compute gradient numerically (finite differences)
        gradient = self._numerical_gradient(counterfactual)

        return total_loss, gradient

    def _numerical_gradient(
        self, counterfactual: np.ndarray, epsilon: float = 1e-5
    ) -> np.ndarray:
        """Compute gradient using finite differences.

        Args:
            counterfactual: Current counterfactual candidate
            epsilon: Small value for finite differences

        Returns:
            Gradient vector
        """
        gradient = np.zeros_like(counterfactual)

        for i in range(len(counterfactual)):
            # Forward difference
            cf_plus = counterfactual.copy()
            cf_plus[i] += epsilon
            loss_plus, _ = self.compute(cf_plus)

            # Backward difference
            cf_minus = counterfactual.copy()
            cf_minus[i] -= epsilon
            loss_minus, _ = self.compute(cf_minus)

            # Central difference
            gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)

        return gradient