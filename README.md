# DeltaExplainer

A production-ready Python implementation of DeltaExplainer: A Software Debugging Approach to Generating Counterfactual Explanations.

## Overview

DeltaExplainer is an innovative approach to generating counterfactual explanations for machine learning models. It combines Delta Debugging (DD) from software engineering with gradient descent optimization to efficiently find minimal feature changes that alter model predictions.

### Key Features

- **Minimal Feature Changes**: Uses Delta Debugging to systematically identify the smallest set of features needed to change predictions
- **Efficient Optimization**: Employs gradient descent with Adam optimizer for finding optimal feature values
- **Model Agnostic**: Works with any ML model that provides predict and predict_proba methods
- **Production Ready**: Clean, modular architecture with comprehensive type hints and configuration management
- **Flexible Distance Metrics**: Supports L1 and L2 distance metrics with MAD normalization

## Installation

```bash
pip install deltaexplainer
```

Or install from source:

```bash
git clone https://github.com/yourusername/deltaexplainer.git
cd deltaexplainer
pip install -e .
```

## Quick Start

```python
from deltaexplainer import DeltaExplainer, DeltaExplainerConfig
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize DeltaExplainer
config = DeltaExplainerConfig(
    max_iterations=5000,
    learning_rate=0.01,
    verbose=True
)
explainer = DeltaExplainer(model, config=config)

# Fit preprocessor on training data
explainer.fit(X_train)

# Generate counterfactual explanation
explanation = explainer.explain(instance)

# View changes
print(explanation.get_changes())
```

## Architecture

DeltaExplainer consists of several modular components:

1. **Delta Debugging Module** (`delta_debugging.py`): Implements the DD algorithm to find minimal feature sets
2. **Optimization Module** (`optimization.py`): Gradient-based optimization with Adam and SGD support
3. **Distance Module** (`distance.py`): Distance metrics and loss function computation
4. **Preprocessing Module** (`preprocessing.py`): Data normalization and feature type handling
5. **Core Module** (`core.py`): Main DeltaExplainer class that orchestrates the explanation process

## Configuration

DeltaExplainer behavior can be customized through `DeltaExplainerConfig`:

```python
config = DeltaExplainerConfig(
    max_iterations=5000,           # Max gradient descent iterations
    learning_rate=0.01,            # Learning rate for optimization
    lambda_param=1.0,              # Weight for distance in loss function
    convergence_threshold=1e-5,    # Convergence threshold
    initial_partitions=2,          # Initial DD partitions
    optimizer="adam",              # Optimizer type: "adam" or "sgd"
    distance_metric="l1",          # Distance metric: "l1" or "l2"
    verbose=False,                 # Print debug information
    seed=42                        # Random seed for reproducibility
)
```

## Algorithm Details

### Delta Debugging

The DD algorithm systematically searches for a minimal set of features:
1. Partitions features into subsets
2. Tests if changing each subset produces desired class change
3. Recursively reduces to find 1-minimal configuration

### Gradient Optimization

Once minimal features are identified, gradient descent optimizes their values:
1. Uses classification loss + distance penalty
2. Applies MAD normalization for robust distance computation
3. Supports both continuous and categorical features

## Requirements

- Python >= 3.8
- numpy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0

## Citation

If you use DeltaExplainer in your research, please cite:

```bibtex
@inproceedings{shree2022deltaexplainer,
  title={DeltaExplainer: A Software Debugging Approach to Generating Counterfactual Explanations},
  author={Shree, Sunny and Chandrasekaran, Jaganmohan and Lei, Yu and Kacker, Raghu N and Kuhn, D Richard},
  booktitle={2022 IEEE International Conference on Artificial Intelligence Testing (AITest)},
  pages={103--110},
  year={2022},
  organization={IEEE}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.