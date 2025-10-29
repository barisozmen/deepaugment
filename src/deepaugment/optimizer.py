"""Bayesian optimization using scikit-optimize - beautiful and configurable."""

import numpy as np
from skopt import Optimizer
from skopt.space import Real, Categorical
from .augment import TRANSFORMS, get_transform_categories


class PolicyOptimizer:
    """
    scikit-optimize based Bayesian optimizer for augmentation policies.

    Beautiful, flexible, powerful. Like Rails for Bayesian optimization.
    """

    def __init__(
        self,
        n_operations=4,
        n_initial=10,
        random_state=42,
        method="bayesian",
        transform_categories=None,
    ):
        """
        Initialize optimizer with sensible defaults and sharp knives.

        Args:
            n_operations: Number of transforms in a policy (default: 4)
            n_initial: Random initial points before optimization (default: 10)
            random_state: Random seed for reproducibility
            method: 'bayesian' or 'random' for search strategy
            transform_categories: List of categories to restrict search space
                                 e.g., ['geometric', 'color'] or None for all
        """
        self.n_ops = n_operations
        self.n_initial = n_initial
        self.method = method
        self.random_state = random_state

        # Transform space - with optional restriction
        if transform_categories:
            cats = get_transform_categories()
            allowed = []
            for cat in transform_categories:
                allowed.extend(cats.get(cat, []))
            self.transform_names = [t for t in TRANSFORMS.keys() if t in allowed]
        else:
            self.transform_names = list(TRANSFORMS.keys())

        self.n_transforms = len(self.transform_names)

        # Build search space for scikit-optimize
        # Each operation has: (transform_name, magnitude)
        dimensions = []
        for _ in range(n_operations):
            dimensions.append(Categorical(range(self.n_transforms), name="transform"))
            dimensions.append(Real(0.0, 1.0, name="magnitude"))

        # Initialize optimizer
        if method == "bayesian":
            # Gaussian Process with Expected Improvement
            self.optimizer = Optimizer(
                dimensions=dimensions,
                base_estimator="RF",  # Random Forest - faster, more robust
                acq_func="EI",  # Expected Improvement
                n_initial_points=n_initial,
                random_state=random_state,
            )
        else:
            # Pure random search
            self.optimizer = None

        # History storage
        self.X = []  # Parameters tried
        self.Y = []  # Scores (we'll negate for minimization)

        # Random state
        np.random.seed(random_state)

    def ask(self):
        """Sample next policy. Convention: random initially, smart after."""
        if self.method == "random":
            return self._random_sample()

        # Ask scikit-optimize for next point
        x = self.optimizer.ask()
        return x

    def _random_sample(self):
        """Random policy - simple and fast."""
        policy = []
        for _ in range(self.n_ops):
            transform_idx = np.random.randint(0, self.n_transforms)
            magnitude = np.random.uniform(0.0, 1.0)
            policy.extend([transform_idx, magnitude])
        return policy

    def tell(self, policy, score):
        """Report result. Simple interface, powerful learning."""
        # Store in history
        self.X.append(policy)
        self.Y.append(score)

        # Tell optimizer (negate score for minimization)
        if self.method == "bayesian":
            self.optimizer.tell(policy, -score)

    def best_policy(self):
        """Get the winner. Returns (policy, score)."""
        if len(self.Y) == 0:
            return None, 0.0

        best_idx = np.argmax(self.Y)
        best_policy = self.X[best_idx]
        best_score = self.Y[best_idx]

        return best_policy, best_score

    def format_policy(self, raw_policy):
        """Convert optimizer format to human-readable policy."""
        policy = []
        for i in range(0, len(raw_policy), 2):
            transform_idx = int(raw_policy[i])
            magnitude = raw_policy[i + 1]
            transform_name = self.transform_names[transform_idx]
            policy.append((transform_name, magnitude))
        return policy

    def get_history(self):
        """Export full history for analysis or resumption."""
        return {
            "X": self.X,
            "Y": self.Y,
            "transform_names": self.transform_names,
            "n_operations": self.n_ops,
        }

    def load_history(self, history):
        """Resume from saved history. Progress over starting fresh!"""
        self.X = history["X"]
        self.Y = history["Y"]

        # Re-tell optimizer about all past evaluations
        if self.method == "bayesian":
            for x, y in zip(self.X, self.Y):
                self.optimizer.tell(x, -y)
