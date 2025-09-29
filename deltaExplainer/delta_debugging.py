"""Delta Debugging algorithm implementation."""

from typing import List, Set, Callable, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class DDResult:
    """Result from Delta Debugging algorithm.

    Attributes:
        minimal_features: Minimal set of feature indices that need to be changed
        iterations: Number of iterations performed
        feature_combinations_tested: Number of feature combinations tested
    """

    minimal_features: Set[int]
    iterations: int
    feature_combinations_tested: int


class DeltaDebugger:
    """Delta Debugging algorithm for finding minimal feature sets."""

    def __init__(self, verbose: bool = False):
        """Initialize DeltaDebugger.

        Args:
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        self.iterations = 0
        self.feature_combinations_tested = 0

    def run(
        self,
        features: List[int],
        test_function: Callable[[Set[int]], bool],
        n: int = 2,
    ) -> DDResult:
        """Run Delta Debugging algorithm.

        Args:
            features: List of feature indices to consider
            test_function: Function that returns True if changing given features
                          produces a different class
            n: Initial number of partitions

        Returns:
            DDResult containing minimal feature set and statistics
        """
        self.iterations = 0
        self.feature_combinations_tested = 0

        minimal_features = self._dd(set(features), test_function, n)

        return DDResult(
            minimal_features=minimal_features,
            iterations=self.iterations,
            feature_combinations_tested=self.feature_combinations_tested,
        )

    def _dd(
        self, features: Set[int], test_function: Callable[[Set[int]], bool], n: int
    ) -> Set[int]:
        """Core Delta Debugging algorithm.

        Args:
            features: Current set of feature indices
            test_function: Function to test if features cause desired change
            n: Number of partitions

        Returns:
            1-minimal set of features
        """
        self.iterations += 1

        if self.verbose:
            print(f"DD iteration {self.iterations}: |features|={len(features)}, n={n}")

        if len(features) == 1:
            return features

        # Create n partitions
        feature_list = list(features)
        partition_size = max(1, len(feature_list) // n)
        partitions = []

        for i in range(n):
            start_idx = i * partition_size
            if i == n - 1:
                # Last partition gets remaining elements
                partition = set(feature_list[start_idx:])
            else:
                end_idx = (i + 1) * partition_size
                partition = set(feature_list[start_idx:end_idx])
            if partition:  # Only add non-empty partitions
                partitions.append(partition)

        # Test each partition
        for partition in partitions:
            self.feature_combinations_tested += 1
            if test_function(partition):
                if self.verbose:
                    print(f"  Found failing subset with {len(partition)} features")
                return self._dd(partition, test_function, 2)

        # Test complements of each partition
        for partition in partitions:
            complement = features - partition
            if complement:
                self.feature_combinations_tested += 1
                if test_function(complement):
                    if self.verbose:
                        print(f"  Found failing complement with {len(complement)} features")
                    return self._dd(complement, test_function, max(n - 1, 2))

        # Increase granularity if possible
        if n < len(features):
            new_n = min(2 * n, len(features))
            if self.verbose:
                print(f"  Increasing granularity to {new_n}")
            return self._dd(features, test_function, new_n)

        # Current configuration is 1-minimal
        if self.verbose:
            print(f"  Found 1-minimal configuration with {len(features)} features")
        return features