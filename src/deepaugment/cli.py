"""
Command-line interface for DeepAugment.

Minimal, elegant CLI using Fire. Convention: sensible defaults from config.
"""

import fire
import numpy as np
from .core import optimize
from .config import DATA_DIR


def cifar10(iterations=None, epochs=None):
    """
    Optimize augmentation for CIFAR-10 dataset.

    Example:
        $ deepaugment cifar10
        $ deepaugment cifar10 --iterations=100 --epochs=15
    """
    try:
        from torchvision.datasets import CIFAR10

        print("📦 Loading CIFAR-10...")
        train_data = CIFAR10(root=str(DATA_DIR), train=True, download=True)
        test_data = CIFAR10(root=str(DATA_DIR), train=False, download=True)

        # Convert to numpy
        X_train = np.array(train_data.data)
        y_train = np.array(train_data.targets)
        X_val = np.array(test_data.data)
        y_val = np.array(test_data.targets)

        print(f"   Train: {X_train.shape}, Val: {X_val.shape}")

        # Run optimization with sensible defaults
        best = optimize(
            X_train,
            y_train,
            X_val,
            y_val,
            iterations=iterations,
            epochs=epochs,
        )

        return best

    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure torchvision is installed: uv add torchvision")
        return None


def fashion_mnist(iterations=None, epochs=None):
    """
    Optimize augmentation for Fashion-MNIST dataset.

    Example:
        $ deepaugment fashion-mnist
        $ deepaugment fashion-mnist --iterations=100
    """
    try:
        from torchvision.datasets import FashionMNIST

        print("📦 Loading Fashion-MNIST...")
        train_data = FashionMNIST(root=str(DATA_DIR), train=True, download=True)
        test_data = FashionMNIST(root=str(DATA_DIR), train=False, download=True)

        # Convert to numpy and add channel dimension (grayscale → RGB)
        X_train = np.array(train_data.data.numpy())[..., None].repeat(3, axis=-1)
        y_train = np.array(train_data.targets.numpy())
        X_val = np.array(test_data.data.numpy())[..., None].repeat(3, axis=-1)
        y_val = np.array(test_data.targets.numpy())

        print(f"   Train: {X_train.shape}, Val: {X_val.shape}")

        # Run optimization
        best = optimize(
            X_train,
            y_train,
            X_val,
            y_val,
            iterations=iterations,
            epochs=epochs,
        )

        return best

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """
    DeepAugment CLI - Find optimal image augmentation policies.

    Minimal interface powered by Fire.
    """
    fire.Fire({
        'cifar10': cifar10,
        'fashion-mnist': fashion_mnist,
    })


if __name__ == '__main__':
    main()
