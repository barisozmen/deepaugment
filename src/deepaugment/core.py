"""
Core DeepAugment API - elegant, minimal, powerful.

Optimized for programmer happiness. Convention over Configuration.
Rails doctrine applied to machine learning.
"""

import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from .config import defaults, EXPERIMENTS_DIR, resolve_device
from .models import create_model
from .policy import PolicySpace
from .search import create_search
from .trainer import evaluate_policy
from .transforms import apply_policy
from .utils import (
    sample_indices,
    split_data,
    save_checkpoint,
    load_checkpoint,
    generate_experiment_name,
    format_policy,
)


# ============================================================
# MAIN API - DeepAugment class
# ============================================================

class DeepAugment:
    """
    Find optimal image augmentation policies using Bayesian optimization.

    Beautiful API inspired by Rails: Convention over Configuration,
    but with sharp knives when you need them.

    Example:
        >>> aug = DeepAugment(X_train, y_train, X_val, y_val)
        >>> best = aug.optimize(iterations=50)
        >>> aug.show_best()
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        # Phase 1: Essential
        model="simple",
        device="auto",
        random_state=None,
        experiment_name=None,
        # Phase 2: Useful
        method="bayesian",
        save_history=None,
        # Phase 3: Advanced
        transform_categories=None,
        custom_reward_fn=None,
        resume_from=None,
        # Core params
        n_operations=None,
        train_size=None,
        val_size=None,
    ):
        """
        Initialize DeepAugment with beautiful defaults and configuration freedom.

        All args are optional - sensible defaults from config.

        Args:
            X_train, y_train: Training data (N, H, W, C) and labels (N,)
            X_val, y_val: Validation data and labels

            # PHASE 1: Essential
            model: Model architecture (default: 'simple')
            device: 'cuda', 'mps', 'cpu', or 'auto' (default: 'auto')
            random_state: Seed for reproducibility (default from config)
            experiment_name: Name for tracking (default: timestamp)

            # PHASE 2: Useful
            method: 'bayesian' or 'random' search (default: 'bayesian')
            save_history: Save optimization history (default from config)

            # PHASE 3: Advanced
            transform_categories: Restrict transforms, e.g., ['geometric', 'color']
            custom_reward_fn: Custom reward function(entry) -> float
            resume_from: Path to checkpoint to continue from

            # Core
            n_operations: Transforms per policy (default from config)
            train_size: Training subset size (default from config)
            val_size: Validation subset size (default from config)
        """
        # Convention: use config defaults
        random_state = random_state or defaults.random_state
        n_operations = n_operations or defaults.n_operations
        train_size = train_size or defaults.train_size
        val_size = val_size or defaults.val_size
        save_history = save_history if save_history is not None else defaults.save_history

        # Store config
        self.model_type = model
        self.device = resolve_device(device)
        self.random_state = random_state
        self.experiment_name = experiment_name or generate_experiment_name()
        self.method = method
        self.save_history = save_history
        self.custom_reward_fn = custom_reward_fn

        # Sample subsets for speed
        train_idx = sample_indices(len(X_train), train_size, seed=random_state)
        val_idx = sample_indices(len(X_val), val_size, seed=random_state)

        self.X_train = X_train[train_idx]
        self.y_train = y_train[train_idx]
        self.X_val = X_val[val_idx]
        self.y_val = y_val[val_idx]
        self.num_classes = len(np.unique(y_train))

        # Policy space
        self.policy_space = PolicySpace(
            n_operations=n_operations,
            transform_categories=transform_categories,
            random_state=random_state,
        )

        # Search strategy
        self.search = create_search(
            method=method,
            policy_space=self.policy_space,
            random_state=random_state,
        )

        # User-facing history (human-readable policies)
        self.history = []
        self.best_score_so_far = 0.0

        # Resume from checkpoint? Progress over starting fresh!
        if resume_from:
            self._resume(resume_from)

    def optimize(
        self,
        iterations=None,
        epochs=None,
        samples=None,
        batch_size=None,
        learning_rate=None,
        early_stopping=None,
        patience=None,
        verbose=True,
    ):
        """
        Search for best augmentation policy.

        All args optional - sensible defaults from config.

        Args:
            iterations: Number of policies to try (default from config)
            epochs: Training epochs per evaluation (default from config)
            samples: Training runs per policy, for averaging (default from config)
            batch_size: Training batch size (default from config)
            learning_rate: Learning rate (default from config)
            early_stopping: Stop if no improvement (default from config)
            patience: Early stopping patience (default from config)
            verbose: Show progress bar

        Returns:
            Best policy as list of (transform, magnitude) tuples
        """
        # Convention: use config defaults
        iterations = iterations or defaults.iterations
        epochs = epochs or defaults.epochs
        samples = samples or defaults.samples
        batch_size = batch_size or defaults.batch_size
        learning_rate = learning_rate or defaults.learning_rate
        early_stopping = early_stopping if early_stopping is not None else defaults.early_stopping
        patience = patience or defaults.patience

        # Progress bar
        iterator = tqdm(range(iterations), desc="Optimizing") if verbose else range(iterations)
        no_improvement = 0

        for i in iterator:
            # Ask search strategy for next policy
            raw_policy = self.search.ask()
            policy = self.policy_space.format_policy(raw_policy)

            # Evaluate policy
            score = evaluate_policy(
                policy=policy,
                train_data=(self.X_train, self.y_train),
                val_data=(self.X_val, self.y_val),
                num_classes=self.num_classes,
                augmenter=apply_policy,
                model_factory=lambda **kw: create_model(self.model_type, **kw),
                epochs=epochs,
                samples=samples,
                device=self.device,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            # Custom reward? Sharp knife for power users
            if self.custom_reward_fn:
                score = self.custom_reward_fn({"policy": policy, "score": score, "iteration": i})

            # Report to search strategy
            self.search.tell(raw_policy, score)

            # Store human-readable history
            self.history.append({"policy": policy, "score": score, "iteration": i})

            # Track improvement
            if score > self.best_score_so_far:
                self.best_score_so_far = score
                no_improvement = 0
            else:
                no_improvement += 1

            # Update progress bar
            if verbose:
                iterator.set_postfix({
                    "best": f"{self.best_score():.3f}",
                    "current": f"{score:.3f}",
                    "no_improve": no_improvement,
                })

            # Early stopping
            if early_stopping and no_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping! No improvement for {patience} iterations.")
                break

            # Auto-save checkpoints
            if self.save_history and (i + 1) % defaults.checkpoint_every == 0:
                self._save_checkpoint()

        # Final save
        if self.save_history:
            self._save_checkpoint(final=True)

        return self.best_policy()

    def best_policy(self):
        """Get best policy found (human-readable)."""
        raw_policy, _ = self.search.best()
        return self.policy_space.format_policy(raw_policy)

    def best_score(self):
        """Get best validation accuracy found."""
        _, score = self.search.best()
        return score

    def show_best(self, n=5):
        """Print top N policies. Beautiful output for programmer happiness."""
        sorted_history = sorted(self.history, key=lambda x: x["score"], reverse=True)

        print(f"\n✨ Top {n} Augmentation Policies")
        print("=" * 70)

        for i, entry in enumerate(sorted_history[:n], 1):
            print(f"\n#{i} | Accuracy: {entry['score']:.3f} | Iteration: {entry['iteration']}")
            print(format_policy(entry["policy"]))

    def _save_checkpoint(self, final=False):
        """Save optimization state for resumption."""
        suffix = "final" if final else "checkpoint"
        filename = f"{self.experiment_name}_{suffix}.json"

        checkpoint = {
            "experiment_name": self.experiment_name,
            "device": self.device,
            "method": self.method,
            "model_type": self.model_type,
            "history": self.history,
            "search_history": self.search.history.to_dict(),
            "best_score": self.best_score(),
            "timestamp": datetime.now().isoformat(),
        }

        save_checkpoint(checkpoint, filename, directory=EXPERIMENTS_DIR)

    def _resume(self, checkpoint_path):
        """Resume from saved checkpoint. Progress over stability!"""
        checkpoint = load_checkpoint(checkpoint_path)

        self.history = checkpoint["history"]
        self.best_score_so_far = checkpoint["best_score"]

        # Restore search strategy history
        if hasattr(self.search, "load_history"):
            self.search.load_history(checkpoint["search_history"])

        print(f"📂 Resumed from {checkpoint_path}")
        print(f"   Best score so far: {self.best_score_so_far:.3f}")


# ============================================================
# QUICK API - One-liner optimization
# ============================================================

def optimize(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    iterations=None,
    epochs=None,
    verbose=True,
    **kwargs,
):
    """
    Quick optimization function for minimal code.

    Convention: if no val set, auto-split 80/20.

    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images (optional, will split if None)
        y_val: Validation labels (optional)
        iterations: Optimization iterations (default from config)
        epochs: Training epochs per policy (default from config)
        verbose: Show progress
        **kwargs: Additional args for DeepAugment

    Returns:
        Best augmentation policy

    Example:
        >>> from deepaugment import optimize
        >>> best = optimize(X, y, iterations=50)
    """
    # Convention: auto-split if needed
    if X_val is None:
        (X_train, y_train), (X_val, y_val) = split_data(X_train, y_train)

    # Create and run
    aug = DeepAugment(X_train, y_train, X_val, y_val, **kwargs)
    best = aug.optimize(iterations=iterations, epochs=epochs, verbose=verbose)

    # Show results
    aug.show_best()

    return best
