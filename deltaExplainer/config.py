"""Configuration management for DeltaExplainer."""

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np


@dataclass
class DeltaExplainerConfig:
    """Configuration for DeltaExplainer.

    Attributes:
        max_iterations: Maximum iterations for gradient descent optimization
        learning_rate: Learning rate for gradient descent
        lambda_param: Weight for distance in loss function
        convergence_threshold: Threshold for convergence check
        initial_partitions: Initial number of partitions for Delta Debugging
        optimizer: Type of optimizer to use
        distance_metric: Type of distance metric
        verbose: Whether to print debug information
        seed: Random seed for reproducibility
    """

    max_iterations: int = 5000
    learning_rate: float = 0.01
    lambda_param: float = 1.0
    convergence_threshold: float = 1e-5
    initial_partitions: int = 2
    optimizer: Literal["adam", "sgd"] = "adam"
    distance_metric: Literal["l1", "l2"] = "l1"
    verbose: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.lambda_param < 0:
            raise ValueError("lambda_param must be non-negative")
        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive")
        if self.initial_partitions < 2:
            raise ValueError("initial_partitions must be at least 2")

        if self.seed is not None:
            np.random.seed(self.seed)