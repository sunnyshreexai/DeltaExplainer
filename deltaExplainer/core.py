"""Core DeltaExplainer implementation."""

from typing import Any, Optional, Union, Dict, List, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .config import DeltaExplainerConfig
from .delta_debugging import DeltaDebugger, DDResult
from .optimization import GradientOptimizer, OptimizationResult
from .distance import LossFunction, DistanceMetrics
from .preprocessing import DataPreprocessor


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation result.

    Attributes:
        original_instance: Original instance
        counterfactual: Generated counterfactual
        original_prediction: Original model prediction
        counterfactual_prediction: Counterfactual model prediction
        changed_features: Indices of changed features
        distance: Distance between original and counterfactual
        iterations: Total iterations performed
        feature_combinations_tested: Number of feature combinations tested
        optimization_converged: Whether optimization converged
    """

    original_instance: Union[np.ndarray, pd.DataFrame]
    counterfactual: Union[np.ndarray, pd.DataFrame]
    original_prediction: Any
    counterfactual_prediction: Any
    changed_features: List[int]
    distance: float
    iterations: int
    feature_combinations_tested: int
    optimization_converged: bool

    def get_changes(self) -> pd.DataFrame:
        """Get a DataFrame showing the changes made.

        Returns:
            DataFrame with original and counterfactual values for changed features
        """
        if isinstance(self.original_instance, pd.DataFrame):
            orig_df = self.original_instance
            cf_df = self.counterfactual
        else:
            orig_df = pd.DataFrame(self.original_instance.reshape(1, -1))
            cf_df = pd.DataFrame(self.counterfactual.reshape(1, -1))

        changes = []
        for idx in self.changed_features:
            if idx < orig_df.shape[1]:
                feature_name = (
                    orig_df.columns[idx]
                    if hasattr(orig_df, "columns")
                    else f"feature_{idx}"
                )
                changes.append(
                    {
                        "feature": feature_name,
                        "original": orig_df.iloc[0, idx],
                        "counterfactual": cf_df.iloc[0, idx],
                    }
                )

        return pd.DataFrame(changes)


class DeltaExplainer:
    """Main DeltaExplainer class for generating counterfactual explanations."""

    def __init__(
        self,
        model: Any,
        preprocessor: Optional[DataPreprocessor] = None,
        config: Optional[DeltaExplainerConfig] = None,
    ):
        """Initialize DeltaExplainer.

        Args:
            model: Trained ML model with predict and predict_proba methods
            preprocessor: Optional preprocessor for data transformation
            config: Configuration object
        """
        self.model = model
        self.preprocessor = preprocessor or DataPreprocessor()
        self.config = config or DeltaExplainerConfig()

        # Validate model
        if not hasattr(model, "predict"):
            raise ValueError("Model must have a predict method")
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model must have a predict_proba method")

        # Initialize components
        self.delta_debugger = DeltaDebugger(verbose=self.config.verbose)
        self.optimizer = GradientOptimizer(
            learning_rate=self.config.learning_rate,
            max_iterations=self.config.max_iterations,
            convergence_threshold=self.config.convergence_threshold,
            optimizer_type=self.config.optimizer,
            verbose=self.config.verbose,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_types: Optional[Dict[int, str]] = None,
    ) -> "DeltaExplainer":
        """Fit the preprocessor on training data.

        Args:
            X: Training data
            feature_types: Optional dict mapping feature index to type

        Returns:
            Self
        """
        self.preprocessor.fit(X, feature_types)
        return self

    def explain(
        self,
        instance: Union[np.ndarray, pd.DataFrame],
        target_class: Optional[int] = None,
    ) -> CounterfactualExplanation:
        """Generate counterfactual explanation for an instance.

        Args:
            instance: Instance to explain
            target_class: Optional target class for counterfactual
                         If None, will target any different class

        Returns:
            CounterfactualExplanation object
        """
        # Ensure instance is 1D
        if isinstance(instance, pd.DataFrame):
            if len(instance) != 1:
                raise ValueError("Instance must be a single row")
            original_instance = instance.copy()
            instance_array = instance.values.flatten()
        else:
            instance_array = np.asarray(instance).flatten()
            original_instance = instance_array.copy()

        # Transform instance
        instance_transformed, continuous_mask, categorical_mask = (
            self.preprocessor.transform(
                instance_array.reshape(1, -1)
                if len(instance_array.shape) == 1
                else instance_array
            )
        )
        instance_transformed = instance_transformed.flatten()

        # Get original prediction
        original_pred = self.model.predict(instance_array.reshape(1, -1))[0]
        original_proba = self.model.predict_proba(instance_array.reshape(1, -1))[0]

        # Create test function for Delta Debugging
        def test_function(feature_indices):
            return self._test_feature_combination(
                instance_transformed,
                feature_indices,
                original_pred,
                target_class,
                continuous_mask,
                categorical_mask,
            )

        # Run Delta Debugging to find minimal feature set
        all_features = list(range(len(instance_transformed)))
        dd_result = self.delta_debugger.run(all_features, test_function)

        if self.config.verbose:
            print(f"Delta Debugging found {len(dd_result.minimal_features)} features to change")

        # Generate counterfactual by optimizing selected features
        counterfactual_transformed = self._optimize_counterfactual(
            instance_transformed,
            dd_result.minimal_features,
            original_pred,
            continuous_mask,
            categorical_mask,
        )

        # Inverse transform counterfactual
        counterfactual = self.preprocessor.inverse_transform(
            counterfactual_transformed.reshape(1, -1)
        )

        # Get counterfactual prediction
        cf_pred = self.model.predict(counterfactual)[0]
        cf_proba = self.model.predict_proba(counterfactual)[0]

        # Calculate distance
        mad = self.preprocessor.get_mad_array()
        distance = DistanceMetrics.total_distance(
            instance_transformed,
            counterfactual_transformed,
            mad,
            continuous_mask,
            categorical_mask,
            self.config.distance_metric,
        )

        # Identify changed features
        changed_features = list(dd_result.minimal_features)

        return CounterfactualExplanation(
            original_instance=original_instance,
            counterfactual=counterfactual,
            original_prediction=original_pred,
            counterfactual_prediction=cf_pred,
            changed_features=changed_features,
            distance=distance,
            iterations=dd_result.iterations,
            feature_combinations_tested=dd_result.feature_combinations_tested,
            optimization_converged=True,  # Will be updated from optimization result
        )

    def _test_feature_combination(
        self,
        instance: np.ndarray,
        feature_indices: set,
        original_class: int,
        target_class: Optional[int],
        continuous_mask: np.ndarray,
        categorical_mask: np.ndarray,
    ) -> bool:
        """Test if changing given features can produce different class.

        Args:
            instance: Transformed instance
            feature_indices: Indices of features to change
            original_class: Original predicted class
            target_class: Target class (optional)
            continuous_mask: Mask for continuous features
            categorical_mask: Mask for categorical features

        Returns:
            True if changing features produces different class
        """
        # Create feature mask
        feature_mask = np.zeros(len(instance), dtype=bool)
        for idx in feature_indices:
            feature_mask[idx] = True

        # Quick optimization to test if features can change class
        result = self._optimize_counterfactual(
            instance,
            feature_indices,
            original_class,
            continuous_mask,
            categorical_mask,
            max_iterations=500,  # Fewer iterations for testing
        )

        # Check if prediction changed
        pred = self.model.predict(result.reshape(1, -1))[0]

        if target_class is not None:
            return pred == target_class
        else:
            return pred != original_class

    def _optimize_counterfactual(
        self,
        instance: np.ndarray,
        feature_indices: set,
        original_class: int,
        continuous_mask: np.ndarray,
        categorical_mask: np.ndarray,
        max_iterations: Optional[int] = None,
    ) -> np.ndarray:
        """Optimize counterfactual for given features.

        Args:
            instance: Transformed instance
            feature_indices: Indices of features to change
            original_class: Original predicted class
            continuous_mask: Mask for continuous features
            categorical_mask: Mask for categorical features
            max_iterations: Optional override for max iterations

        Returns:
            Optimized counterfactual
        """
        # Create feature mask
        feature_mask = np.zeros(len(instance), dtype=bool)
        for idx in feature_indices:
            feature_mask[idx] = True

        # Create loss function
        mad = self.preprocessor.get_mad_array()
        loss_fn = LossFunction(
            model_predict_proba=lambda x: self.model.predict_proba(
                self.preprocessor.inverse_transform(x)
            ),
            original_instance=instance,
            original_class=original_class,
            mad=mad,
            continuous_mask=continuous_mask,
            categorical_mask=categorical_mask,
            lambda_param=self.config.lambda_param,
            distance_metric=self.config.distance_metric,
        )

        # Override max iterations if specified
        if max_iterations is not None:
            original_max = self.optimizer.max_iterations
            self.optimizer.max_iterations = max_iterations

        # Optimize
        result = self.optimizer.optimize(
            initial_point=instance.copy(),
            loss_function=loss_fn.compute,
            feature_mask=feature_mask,
        )

        # Restore original max iterations
        if max_iterations is not None:
            self.optimizer.max_iterations = original_max

        return result.counterfactual