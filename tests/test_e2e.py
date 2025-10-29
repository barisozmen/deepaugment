"""
End-to-end tests for DeepAugment.

DHH style: Real scenarios, no mocking, fast and confident.
Tests the whole stack from API to optimization.
"""

import numpy as np
import pytest
from pathlib import Path
from deepaugment import DeepAugment, optimize, apply_policy, TRANSFORMS


# Fixtures - Convention over Configuration
@pytest.fixture
def tiny_dataset():
    """Tiny synthetic dataset for fast testing."""
    np.random.seed(42)
    X_train = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.randint(0, 5, 100)
    X_val = np.random.randint(0, 255, (50, 32, 32, 3), dtype=np.uint8)
    y_val = np.random.randint(0, 5, 50)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def sample_image():
    """Single image for transform testing."""
    return np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)


# E2E Test 1: Full optimization workflow
def test_full_optimization_workflow(tiny_dataset):
    """
    E2E: Initialize → Optimize → Get results

    This is the happy path - what 80% of users will do.
    """
    X_train, y_train, X_val, y_val = tiny_dataset

    # Initialize
    aug = DeepAugment(
        X_train, y_train, X_val, y_val, train_size=50, val_size=25, random_state=42
    )

    # Optimize (tiny run for speed)
    best = aug.optimize(iterations=3, epochs=2, verbose=False)

    # Assertions
    assert best is not None
    assert len(best) == 4  # n_operations=4 by default
    assert all(name in TRANSFORMS for name, _ in best)
    assert all(0 <= mag <= 1 for _, mag in best)
    assert aug.best_score() > 0


# E2E Test 2: Quick optimize() function
def test_quick_optimize_function(tiny_dataset):
    """
    E2E: One-liner optimization

    For users who just want results fast.
    """
    X_train, y_train, _, _ = tiny_dataset

    # Convention: auto-split train/val
    best = optimize(
        X_train,
        y_train,
        iterations=2,
        epochs=2,
        train_size=40,
        val_size=20,
        verbose=False,
    )

    assert best is not None
    assert len(best) > 0


# E2E Test 3: Configuration options (Phase 1, 2, 3)
def test_configuration_options(tiny_dataset):
    """
    E2E: All configurations work together
    """
    X_train, y_train, X_val, y_val = tiny_dataset

    # Custom reward function
    def reward_fn(entry):
        return entry["score"] * 1.1  # Bonus for being awesome

    aug = DeepAugment(
        X_train,
        y_train,
        X_val,
        y_val,
        # Phase 1
        device="cpu",
        random_state=123,
        experiment_name="test_exp",
        # Phase 2
        method="bayesian",
        save_history=False,  # Don't clutter
        # Phase 3
        transform_categories=["geometric", "color"],
        custom_reward_fn=reward_fn,
        # Core
        train_size=40,
        val_size=20,
        n_operations=2,
    )

    best = aug.optimize(
        iterations=2, epochs=2, early_stopping=True, patience=5, verbose=False
    )

    assert best is not None
    assert len(best) == 2  # n_operations=2


# E2E Test 4: Random search method
def test_random_search_method(tiny_dataset):
    """
    E2E: Random search as alternative to Bayesian
    """
    X_train, y_train, X_val, y_val = tiny_dataset

    aug = DeepAugment(
        X_train, y_train, X_val, y_val, method="random", train_size=30, val_size=15
    )

    best = aug.optimize(iterations=2, epochs=2, verbose=False)

    assert best is not None


# E2E Test 5: Transform categories restriction
def test_transform_categories(tiny_dataset):
    """
    E2E: Restrict search space to specific transform types

    Power user feature for domain-specific optimization.
    """
    X_train, y_train, X_val, y_val = tiny_dataset

    aug = DeepAugment(
        X_train,
        y_train,
        X_val,
        y_val,
        transform_categories=["geometric"],  # Only geometric transforms
        train_size=30,
        val_size=15,
        n_operations=2,
    )

    best = aug.optimize(iterations=2, epochs=1, verbose=False)

    # Verify only geometric transforms in policy
    geometric_transforms = [
        "rotate",
        "flip_h",
        "flip_v",
        "affine",
        "shear",
        "perspective",
        "elastic",
        "random_crop",
    ]
    for transform_name, _ in best:
        assert transform_name in geometric_transforms


# E2E Test 6: Resume optimization
def test_resume_optimization(tiny_dataset, tmp_path):
    """
    E2E: Save and resume optimization
    """
    X_train, y_train, X_val, y_val = tiny_dataset

    # First run
    aug1 = DeepAugment(
        X_train,
        y_train,
        X_val,
        y_val,
        experiment_name="resume_test",
        save_history=True,
        train_size=30,
        val_size=15,
    )
    aug1.optimize(iterations=2, epochs=1, verbose=False)

    # Find checkpoint
    checkpoint_files = list(Path("experiments").glob("resume_test_*.json"))
    assert len(checkpoint_files) > 0

    checkpoint_path = checkpoint_files[0]

    # Resume from checkpoint
    aug2 = DeepAugment(
        X_train,
        y_train,
        X_val,
        y_val,
        resume_from=checkpoint_path,
        train_size=30,
        val_size=15,
    )

    assert len(aug2.history) > 0  # Has resumed history

    # Cleanup
    for f in checkpoint_files:
        f.unlink()


# E2E Test 7: Apply policy to images
def test_apply_policy_e2e(sample_image):
    """
    E2E: Apply discovered policy to new images

    The final goal: augment your actual data.
    """
    # Simple policy
    policy = [("rotate", 0.3), ("brightness", 0.2), ("flip_h", 0.5)]

    # Apply
    augmented = apply_policy(sample_image, policy)

    assert augmented.shape == sample_image.shape
    assert augmented.dtype == sample_image.dtype


# E2E Test 8: All transforms work
def test_all_transforms_work(sample_image):
    """
    E2E: Every transform in TRANSFORMS can be applied
    """
    for transform_name in TRANSFORMS.keys():
        policy = [(transform_name, 0.5)]

        try:
            augmented = apply_policy(sample_image, policy)
            assert augmented.shape == sample_image.shape
        except Exception as e:
            pytest.fail(f"Transform '{transform_name}' failed: {e}")


# E2E Test 9: Device auto-detection
def test_device_auto_detection(tiny_dataset):
    """
    E2E: Auto device selection works
    """
    X_train, y_train, X_val, y_val = tiny_dataset

    aug = DeepAugment(
        X_train,
        y_train,
        X_val,
        y_val,
        device="auto",  # Magic!
        train_size=20,
        val_size=10,
    )

    assert aug.device in ["cuda", "mps", "cpu"]


# E2E Test 10: Beautiful output
def test_show_best_output(tiny_dataset, capsys):
    """
    E2E: show_best() produces beautiful, readable output
    """
    X_train, y_train, X_val, y_val = tiny_dataset

    aug = DeepAugment(X_train, y_train, X_val, y_val, train_size=30, val_size=15)
    aug.optimize(iterations=2, epochs=1, verbose=False)

    # Capture output
    aug.show_best(n=2)
    captured = capsys.readouterr()

    # Check for beautiful formatting
    assert "✨" in captured.out or "Top" in captured.out
    assert "Accuracy" in captured.out
    assert "magnitude" in captured.out
