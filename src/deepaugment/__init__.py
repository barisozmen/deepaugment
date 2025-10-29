"""
DeepAugment: automated data augmentation using Bayesian optimization.

Find optimal augmentation policies for your image dataset automatically.
"""

from .core import DeepAugment, optimize
from .augment import TRANSFORMS, apply_policy, create_augmenter

__version__ = "2.0.9"
__all__ = ["DeepAugment", "optimize", "TRANSFORMS", "apply_policy", "create_augmenter"]
