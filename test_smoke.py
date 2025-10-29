"""Smoke test to verify DeepAugment works."""

import numpy as np
from deepaugment import DeepAugment, optimize, TRANSFORMS, apply_policy


def test_imports():
    """Test that all imports work."""
    assert DeepAugment is not None
    assert optimize is not None
    print(f"✓ Imports work ({len(TRANSFORMS)} transforms)")


def test_apply_policy():
    """Test applying a policy to an image."""
    # Create dummy image
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

    # Create simple policy
    policy = [('rotate', 0.3), ('brightness', 0.2)]

    # Apply policy
    augmented = apply_policy(image, policy)

    assert augmented.shape == image.shape
    print("✓ Apply policy works")


def test_deepaugment_init():
    """Test DeepAugment initialization."""
    # Create tiny dummy dataset
    X_train = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.randint(0, 10, 100)
    X_val = np.random.randint(0, 255, (50, 32, 32, 3), dtype=np.uint8)
    y_val = np.random.randint(0, 10, 50)

    # Initialize
    aug = DeepAugment(X_train, y_train, X_val, y_val, train_size=50, val_size=25)

    assert aug.X_train.shape[0] == 50
    assert aug.X_val.shape[0] == 25
    print("✓ DeepAugment initialization works")


if __name__ == '__main__':
    print("Running smoke tests...")
    test_imports()
    test_apply_policy()
    test_deepaugment_init()
    print("\n✨ All smoke tests passed!")
