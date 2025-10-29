"""
Policy operations - elegant, composable, functional.

A policy is a list of (transform_name, magnitude) tuples.
Everything related to policy representation lives here (SSOT).
"""

import numpy as np
from attrs import define, field
from .transforms import get_transform_names
from .utils import format_policy, format_policy_summary


# ============================================================
# POLICY REPRESENTATION
# ============================================================

@define
class PolicySpace:
    """
    Policy search space. Knows everything about what policies can be.

    Follows SSOT: all policy space logic lives here.
    """

    n_operations: int = field(default=4)
    transform_categories: list = field(default=None)
    random_state: int = field(default=42)

    # Derived attributes
    transform_names: list = field(init=False)
    n_transforms: int = field(init=False)
    n_dimensions: int = field(init=False)

    def __attrs_post_init__(self):
        """Initialize derived attributes. Convention: computed, not stored."""
        self.transform_names = get_transform_names(self.transform_categories)
        self.n_transforms = len(self.transform_names)
        self.n_dimensions = self.n_operations * 2  # Each op: (transform_idx, magnitude)

        # Set random seed
        np.random.seed(self.random_state)

    def random_policy(self):
        """Sample random policy from space."""
        raw = []
        for _ in range(self.n_operations):
            transform_idx = np.random.randint(0, self.n_transforms)
            magnitude = np.random.uniform(0.0, 1.0)
            raw.extend([transform_idx, magnitude])
        return raw

    def format_policy(self, raw_policy):
        """
        Convert raw policy (optimizer format) to human-readable.

        Raw: [idx1, mag1, idx2, mag2, ...]
        Human: [(name1, mag1), (name2, mag2), ...]
        """
        policy = []
        for i in range(0, len(raw_policy), 2):
            transform_idx = int(raw_policy[i])
            magnitude = raw_policy[i + 1]
            transform_name = self.transform_names[transform_idx]
            policy.append((transform_name, magnitude))
        return policy

    def dimensions(self):
        """
        Get search space dimensions for optimizer.

        Returns list of (min, max) or categorical values for each dimension.
        """
        from skopt.space import Real, Categorical

        dims = []
        for _ in range(self.n_operations):
            dims.append(Categorical(range(self.n_transforms), name="transform"))
            dims.append(Real(0.0, 1.0, name="magnitude"))
        return dims


# ============================================================
# POLICY EVALUATION HISTORY
# ============================================================

@define
class PolicyHistory:
    """
    Track evaluated policies and scores.

    Minimal, elegant storage. Unix: do one thing well.
    """

    policies: list = field(factory=list)  # Raw policies
    scores: list = field(factory=list)    # Corresponding scores

    def add(self, policy, score):
        """Add evaluation result."""
        self.policies.append(policy)
        self.scores.append(score)

    def best(self):
        """Get best policy and score. Simple."""
        if not self.scores:
            return None, 0.0

        best_idx = np.argmax(self.scores)
        return self.policies[best_idx], self.scores[best_idx]

    def top_k(self, k=5):
        """Get top K policies."""
        if not self.scores:
            return []

        indices = np.argsort(self.scores)[::-1][:k]
        return [(self.policies[i], self.scores[i]) for i in indices]

    def to_dict(self):
        """Export as dict for serialization."""
        return {
            "policies": self.policies,
            "scores": self.scores,
        }

    @classmethod
    def from_dict(cls, data):
        """Load from dict."""
        return cls(
            policies=data["policies"],
            scores=data["scores"],
        )

    def __len__(self):
        return len(self.scores)


# ============================================================
# HELPERS
# ============================================================

def display_policy(policy, indent=4):
    """Pretty print policy. For programmer happiness."""
    return format_policy(policy, indent=indent)


def policy_summary(policy):
    """One-line policy summary."""
    return format_policy_summary(policy)
