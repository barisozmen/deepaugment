"""Core DeepAugment API - elegant and minimal."""

import numpy as np
from tqdm import tqdm
from .optimizer import PolicyOptimizer
from .model import SimpleCNN, train_model, evaluate_policy
from .augment import apply_policy


class DeepAugment:
    """
    Find optimal image augmentation policies using Bayesian optimization.

    Example:
        >>> aug = DeepAugment(X_train, y_train, X_val, y_val)
        >>> best_policy = aug.optimize(iterations=50)
        >>> aug.show_best()
    """

    def __init__(self, X_train, y_train, X_val, y_val, n_operations=4, train_size=2000, val_size=500):
        """
        Initialize DeepAugment.

        Args:
            X_train: Training images (N, H, W, C)
            y_train: Training labels (N,)
            X_val: Validation images (N, H, W, C)
            y_val: Validation labels (N,)
            n_operations: Number of transforms per policy
            train_size: Subset size for training (for speed)
            val_size: Subset size for validation
        """
        # Sample subsets for faster iteration
        train_idx = np.random.choice(len(X_train), min(train_size, len(X_train)), replace=False)
        val_idx = np.random.choice(len(X_val), min(val_size, len(X_val)), replace=False)

        self.X_train = X_train[train_idx]
        self.y_train = y_train[train_idx]
        self.X_val = X_val[val_idx]
        self.y_val = y_val[val_idx]
        self.num_classes = len(np.unique(y_train))

        self.optimizer = PolicyOptimizer(n_operations=n_operations)
        self.history = []

    def optimize(self, iterations=50, epochs=10, samples=1, verbose=True):
        """
        Search for best augmentation policy.

        Args:
            iterations: Number of policies to try
            epochs: Training epochs per evaluation
            samples: Number of training runs per policy (averaged)
            verbose: Show progress bar

        Returns:
            Best policy as list of (transform, magnitude) tuples
        """
        iterator = tqdm(range(iterations), desc="Optimizing") if verbose else range(iterations)

        for i in iterator:
            # Sample next policy
            raw_policy = self.optimizer.ask()
            policy = self.optimizer.format_policy(raw_policy)

            # Evaluate it
            score = evaluate_policy(
                policy,
                (self.X_train, self.y_train),
                (self.X_val, self.y_val),
                self.num_classes,
                apply_policy,
                epochs=epochs,
                samples=samples,
            )

            # Report result
            self.optimizer.tell(raw_policy, score)
            self.history.append({'policy': policy, 'score': score})

            if verbose:
                iterator.set_postfix({'best': f'{self.best_score():.3f}', 'current': f'{score:.3f}'})

        return self.best_policy()

    def best_policy(self):
        """Get best policy found."""
        raw_policy, _ = self.optimizer.best_policy()
        return self.optimizer.format_policy(raw_policy)

    def best_score(self):
        """Get best validation accuracy found."""
        _, score = self.optimizer.best_policy()
        return score

    def show_best(self, n=5):
        """Print top N policies."""
        sorted_history = sorted(self.history, key=lambda x: x['score'], reverse=True)
        print(f"\nTop {n} Policies:")
        print("=" * 60)
        for i, entry in enumerate(sorted_history[:n], 1):
            print(f"\n#{i} - Accuracy: {entry['score']:.3f}")
            for transform, magnitude in entry['policy']:
                print(f"  {transform:20s} magnitude={magnitude:.2f}")


def optimize(X_train, y_train, X_val=None, y_val=None, iterations=50, **kwargs):
    """
    Quick optimization function.

    If no validation set provided, splits training data 80/20.

    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images (optional)
        y_val: Validation labels (optional)
        iterations: Optimization iterations
        **kwargs: Additional args for DeepAugment

    Returns:
        Best augmentation policy
    """
    if X_val is None:
        split = int(0.8 * len(X_train))
        X_train, X_val = X_train[:split], X_train[split:]
        y_train, y_val = y_train[:split], y_train[split:]

    aug = DeepAugment(X_train, y_train, X_val, y_val, **kwargs)
    best = aug.optimize(iterations=iterations)
    aug.show_best()
    return best
