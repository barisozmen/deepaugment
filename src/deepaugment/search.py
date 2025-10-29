"""
Search strategies - Bayesian optimization and random search.

Minimal, composable, elegant. Strategy pattern without the boilerplate.
"""

import numpy as np
from skopt import Optimizer
from attrs import define
from .policy import PolicySpace, PolicyHistory
from .config import defaults


# ============================================================
# SEARCH STRATEGY BASE
# ============================================================

@define
class SearchStrategy:
    """
    Base for search strategies. Minimal interface.

    ask() → get next policy to try
    tell() → report result
    """

    policy_space: PolicySpace
    history: PolicyHistory = None

    def __attrs_post_init__(self):
        """Initialize history if not provided."""
        if self.history is None:
            self.history = PolicyHistory()

    def ask(self):
        """Get next policy to try. Subclasses implement."""
        raise NotImplementedError

    def tell(self, policy, score):
        """Report evaluation result. Updates history."""
        self.history.add(policy, score)

    def best(self):
        """Get best policy found so far."""
        return self.history.best()


# ============================================================
# RANDOM SEARCH - Baseline strategy
# ============================================================

@define
class RandomSearch(SearchStrategy):
    """
    Random search baseline. Simple, fast, surprisingly effective.

    No learning, just exploration. Perfect for sanity checks.
    """

    def ask(self):
        """Sample random policy."""
        return self.policy_space.random_policy()


# ============================================================
# BAYESIAN OPTIMIZATION - Smart search
# ============================================================

@define
class BayesianSearch(SearchStrategy):
    """
    Bayesian optimization using scikit-optimize.

    Learn from past evaluations to suggest better policies.
    Uses Random Forest + Expected Improvement.
    """

    n_initial: int = None
    random_state: int = None

    # Internal optimizer (scikit-optimize)
    _optimizer: Optimizer = None

    def __attrs_post_init__(self):
        """Initialize Bayesian optimizer."""
        super().__attrs_post_init__()

        # Convention: use defaults from config
        self.n_initial = self.n_initial or defaults.n_initial_points
        self.random_state = self.random_state or defaults.random_state

        # Build scikit-optimize optimizer
        self._optimizer = Optimizer(
            dimensions=self.policy_space.dimensions(),
            base_estimator="RF",       # Random Forest - robust and fast
            acq_func="EI",              # Expected Improvement
            n_initial_points=self.n_initial,
            random_state=self.random_state,
        )

    def ask(self):
        """
        Ask optimizer for next policy.

        Convention: random initially, smart after.
        """
        return self._optimizer.ask()

    def tell(self, policy, score):
        """
        Report result to optimizer.

        scikit-optimize minimizes, so we negate the score.
        """
        super().tell(policy, score)
        self._optimizer.tell(policy, -score)  # Negate: maximize → minimize

    def load_history(self, history_data):
        """Resume from saved history. Progress over stability!"""
        hist = PolicyHistory.from_dict(history_data)

        # Re-tell optimizer about past evaluations
        for policy, score in zip(hist.policies, hist.scores):
            self._optimizer.tell(policy, -score)

        self.history = hist


# ============================================================
# FACTORY - Convention over Configuration
# ============================================================

def create_search(
    method="bayesian",
    policy_space=None,
    n_initial=None,
    random_state=None,
):
    """
    Create search strategy by name.

    Convention: 'bayesian' is default, 'random' is baseline.
    """
    # Create policy space if not provided
    if policy_space is None:
        policy_space = PolicySpace(random_state=random_state or defaults.random_state)

    if method == "bayesian":
        return BayesianSearch(
            policy_space=policy_space,
            n_initial=n_initial,
            random_state=random_state,
        )
    elif method == "random":
        return RandomSearch(policy_space=policy_space)
    else:
        raise ValueError(f"Unknown search method: {method}. Use 'bayesian' or 'random'.")
