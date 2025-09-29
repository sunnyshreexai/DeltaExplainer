"""Gradient descent optimization for counterfactual generation."""

from typing import Optional, Dict, Any, Callable, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Result from optimization process.

    Attributes:
        counterfactual: Final counterfactual example
        loss_history: History of loss values
        iterations: Number of iterations performed
        converged: Whether optimization converged
    """

    counterfactual: np.ndarray
    loss_history: list
    iterations: int
    converged: bool


class GradientOptimizer:
    """Gradient descent optimizer for counterfactual generation."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 5000,
        convergence_threshold: float = 1e-5,
        optimizer_type: str = "adam",
        verbose: bool = False,
    ):
        """Initialize optimizer.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence
            optimizer_type: Type of optimizer ('adam' or 'sgd')
            verbose: Whether to print debug information
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.optimizer_type = optimizer_type
        self.verbose = verbose

        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def optimize(
        self,
        initial_point: np.ndarray,
        loss_function: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        feature_mask: np.ndarray,
        bounds: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Optimize counterfactual using gradient descent.

        Args:
            initial_point: Initial counterfactual candidate
            loss_function: Function that returns (loss, gradient)
            feature_mask: Binary mask indicating which features can be changed
            bounds: Optional bounds for features

        Returns:
            OptimizationResult containing final counterfactual and statistics
        """
        current_point = initial_point.copy()
        loss_history = []

        # Initialize optimizer state
        if self.optimizer_type == "adam":
            m = np.zeros_like(current_point)  # First moment estimate
            v = np.zeros_like(current_point)  # Second moment estimate
            t = 0  # Time step

        converged = False
        iteration = 0

        for iteration in range(self.max_iterations):
            # Compute loss and gradient
            loss, gradient = loss_function(current_point)
            loss_history.append(loss)

            # Apply feature mask to gradient
            gradient = gradient * feature_mask

            # Check convergence
            if len(loss_history) > 1:
                loss_change = abs(loss_history[-2] - loss_history[-1])
                if loss_change < self.convergence_threshold:
                    converged = True
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break

            # Update parameters based on optimizer type
            if self.optimizer_type == "adam":
                t += 1
                # Update biased first moment estimate
                m = self.beta1 * m + (1 - self.beta1) * gradient
                # Update biased second moment estimate
                v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - self.beta1 ** t)
                # Compute bias-corrected second moment estimate
                v_hat = v / (1 - self.beta2 ** t)
                # Update parameters
                current_point -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:  # sgd
                current_point -= self.learning_rate * gradient

            # Apply bounds if specified
            if bounds:
                for idx, (lower, upper) in bounds.items():
                    current_point[idx] = np.clip(current_point[idx], lower, upper)

            # Ensure values stay in [0, 1] range for normalized features
            current_point = np.clip(current_point, 0, 1)

            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6f}")

        return OptimizationResult(
            counterfactual=current_point,
            loss_history=loss_history,
            iterations=iteration + 1,
            converged=converged,
        )