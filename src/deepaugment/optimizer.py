"""Bayesian optimization for augmentation policy search."""

import numpy as np
from skopt import Optimizer
from skopt.space import Real, Categorical
from .augment import TRANSFORMS


def create_policy_space(n_operations=4):
    """
    Create search space for augmentation policies.

    Each operation has: [transform_type, magnitude]
    Policy has n_operations such pairs.
    """
    transform_names = list(TRANSFORMS.keys())
    space = []
    for i in range(n_operations):
        space.append(Categorical(transform_names, name=f'op{i}_type'))
        space.append(Real(0.0, 1.0, name=f'op{i}_magnitude'))
    return space


class PolicyOptimizer:
    """Bayesian optimizer for finding best augmentation policies."""

    def __init__(self, n_operations=4, n_initial=10, random_state=42):
        """
        Initialize optimizer.

        Args:
            n_operations: Number of transforms in a policy
            n_initial: Number of random initial points before optimization
            random_state: Random seed for reproducibility
        """
        self.n_ops = n_operations
        self.space = create_policy_space(n_operations)
        self.opt = Optimizer(
            dimensions=self.space,
            n_initial_points=n_initial,
            base_estimator='RF',  # Random Forest
            acq_func='EI',  # Expected Improvement
            random_state=random_state,
        )
        self.history = []

    def ask(self):
        """Sample next policy to try."""
        return self.opt.ask()

    def tell(self, policy, score):
        """Report result of trying a policy."""
        # Optimizer minimizes, so negate score (we want to maximize accuracy)
        self.opt.tell(policy, -score)
        self.history.append({'policy': policy, 'score': score})

    def best_policy(self):
        """Get best policy found so far."""
        if not self.history:
            return None, 0.0
        best = max(self.history, key=lambda x: x['score'])
        return best['policy'], best['score']

    def format_policy(self, raw_policy):
        """Convert optimizer format to readable policy."""
        return [(raw_policy[i], raw_policy[i+1])
                for i in range(0, len(raw_policy), 2)]
