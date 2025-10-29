"""
DeepAugment: automated data augmentation using Bayesian optimization.

Find optimal augmentation policies for your image dataset automatically.
"""

from importlib.metadata import version
__version__ = version("deepaugment")


# Core API
from .core import DeepAugment, optimize

# Transforms
from .transforms import TRANSFORMS, apply_policy, create_augmenter

# For power users who want direct access
from .models import create_model, SimpleCNN
from .policy import PolicySpace
from .search import create_search
from .trainer import train_model, evaluate_policy
from .config import defaults

__all__ = [
    # Main API
    "DeepAugment",
    "optimize",
    # Transforms
    "TRANSFORMS",
    "apply_policy",
    "create_augmenter",
    # Advanced (for power users)
    "create_model",
    "SimpleCNN",
    "PolicySpace",
    "create_search",
    "train_model",
    "evaluate_policy",
    "defaults",
]
