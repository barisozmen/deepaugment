"""Core DeepAugment API - elegant, configurable, and powerful."""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from .optimizer import PolicyOptimizer
from .model import evaluate_policy
from .augment import apply_policy


def auto_device():
    """Convention: auto-detect best device. CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        random_state=42,
        experiment_name=None,
        # Phase 2: Useful
        method="bayesian",
        save_history=True,
        # Phase 3: Advanced
        transform_categories=None,
        custom_reward_fn=None,
        resume_from=None,
        # Core params
        n_operations=4,
        train_size=2000,
        val_size=500,
    ):
        """
        Initialize DeepAugment with beautiful defaults and configuration freedom.

        Args:
            X_train, y_train: Training data (N, H, W, C) and labels (N,)
            X_val, y_val: Validation data and labels

            # PHASE 1: Essential
            model: 'simple' or custom nn.Module (default: 'simple')
            device: 'cuda', 'mps', 'cpu', or 'auto' (default: 'auto')
            random_state: Seed for reproducibility (default: 42)
            experiment_name: Name for tracking (default: timestamp)

            # PHASE 2: Useful
            method: 'bayesian' or 'random' search (default: 'bayesian')
            save_history: Save optimization history to JSON (default: True)

            # PHASE 3: Advanced
            transform_categories: Restrict search space, e.g., ['geometric', 'color']
            custom_reward_fn: Custom reward function(history) -> float
            resume_from: Path to saved history JSON to continue optimization

            # Core
            n_operations: Transforms per policy (default: 4)
            train_size: Training subset size (default: 2000)
            val_size: Validation subset size (default: 500)
        """
        # Convention: smart defaults
        self.device = auto_device() if device == "auto" else device
        self.model_type = model
        self.random_state = random_state
        self.experiment_name = experiment_name or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.method = method
        self.save_history = save_history
        self.custom_reward_fn = custom_reward_fn
        self.transform_categories = transform_categories

        # Sample subsets for speed
        np.random.seed(random_state)
        train_idx = np.random.choice(
            len(X_train), min(train_size, len(X_train)), replace=False
        )
        val_idx = np.random.choice(len(X_val), min(val_size, len(X_val)), replace=False)

        self.X_train = X_train[train_idx]
        self.y_train = y_train[train_idx]
        self.X_val = X_val[val_idx]
        self.y_val = y_val[val_idx]
        self.num_classes = len(np.unique(y_train))

        # History storage - initialize first
        self.history = []
        self.best_score_so_far = 0.0

        # Optimizer with all the bells and whistles
        self.optimizer = PolicyOptimizer(
            n_operations=n_operations,
            random_state=random_state,
            method=method,
            transform_categories=transform_categories,
        )

        # Resume from checkpoint? Progress over starting fresh!
        if resume_from:
            self._resume_optimization(resume_from)

    def optimize(
        self,
        iterations=50,
        epochs=10,
        samples=1,
        batch_size=64,
        learning_rate=0.001,
        early_stopping=False,
        patience=10,
        verbose=True,
    ):
        """
        Search for best augmentation policy.

        Args:
            iterations: Number of policies to try
            epochs: Training epochs per evaluation
            samples: Number of training runs per policy (averaged)
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            early_stopping: Stop if no improvement (Phase 2)
            patience: Early stopping patience
            verbose: Show progress bar

        Returns:
            Best policy as list of (transform, magnitude) tuples
        """
        iterator = (
            tqdm(range(iterations), desc="Optimizing") if verbose else range(iterations)
        )
        no_improvement_count = 0

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
                device=self.device,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            # Custom reward? Sharp knife for power users
            if self.custom_reward_fn:
                score = self.custom_reward_fn(
                    {"policy": policy, "score": score, "iteration": i}
                )

            # Report result
            self.optimizer.tell(raw_policy, score)
            self.history.append({"policy": policy, "score": score, "iteration": i})

            # Track improvement for early stopping
            if score > self.best_score_so_far:
                self.best_score_so_far = score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if verbose:
                iterator.set_postfix(
                    {
                        "best": f"{self.best_score():.3f}",
                        "current": f"{score:.3f}",
                        "no_improve": no_improvement_count,
                    }
                )

            # Early stopping? Optimize for happiness by not wasting time
            if early_stopping and no_improvement_count >= patience:
                if verbose:
                    print(
                        f"\nEarly stopping! No improvement for {patience} iterations."
                    )
                break

            # Auto-save progress
            if self.save_history and (i + 1) % 10 == 0:
                self._save_checkpoint()

        # Final save
        if self.save_history:
            self._save_checkpoint(final=True)

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
        """Print top N policies. Beautiful output."""
        sorted_history = sorted(self.history, key=lambda x: x["score"], reverse=True)
        print(f"\n✨ Top {n} Augmentation Policies")
        print("=" * 70)
        for i, entry in enumerate(sorted_history[:n], 1):
            print(
                f"\n#{i} | Accuracy: {entry['score']:.3f} | Iteration: {entry['iteration']}"
            )
            for transform, magnitude in entry["policy"]:
                print(f"     {transform:20s} magnitude={magnitude:.3f}")

    def _save_checkpoint(self, final=False):
        """Save optimization history for resumption."""
        Path("experiments").mkdir(exist_ok=True)
        filename = f"experiments/{self.experiment_name}_{'final' if final else 'checkpoint'}.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            return obj

        checkpoint = {
            "experiment_name": self.experiment_name,
            "device": self.device,
            "method": self.method,
            "history": convert_to_native(self.history),
            "optimizer_state": convert_to_native(self.optimizer.get_history()),
            "best_score": float(self.best_score()),
            "timestamp": datetime.now().isoformat(),
        }

        with open(filename, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _resume_optimization(self, resume_path):
        """Resume from saved checkpoint. Progress over stability!"""
        with open(resume_path) as f:
            checkpoint = json.load(f)

        self.history = checkpoint["history"]
        self.optimizer.load_history(checkpoint["optimizer_state"])
        self.best_score_so_far = checkpoint["best_score"]
        print(f"📂 Resumed from {resume_path}")
        print(f"   Best score so far: {self.best_score_so_far:.3f}")


def optimize(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    iterations=50,
    epochs=10,
    verbose=True,
    **kwargs,
):
    """
    Quick optimization function. Convention: if no val set, auto-split 80/20.

    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images (optional, will split if None)
        y_val: Validation labels (optional)
        iterations: Optimization iterations
        epochs: Training epochs per policy evaluation
        verbose: Show progress bar
        **kwargs: Additional args for DeepAugment

    Returns:
        Best augmentation policy
    """
    # Convention: auto-split if needed
    if X_val is None:
        split = int(0.8 * len(X_train))
        X_train, X_val = X_train[:split], X_train[split:]
        y_train, y_val = y_train[:split], y_train[split:]

    aug = DeepAugment(X_train, y_train, X_val, y_val, **kwargs)
    best = aug.optimize(iterations=iterations, epochs=epochs, verbose=verbose)
    aug.show_best()
    return best
