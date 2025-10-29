"""Command-line interface for DeepAugment."""

import fire
import numpy as np


def cifar10(iterations=50, epochs=10):
    """Optimize augmentation for CIFAR-10 dataset."""
    from .core import DeepAugment

    # Load CIFAR-10
    try:
        from torchvision.datasets import CIFAR10

        print("Loading CIFAR-10...")
        train_data = CIFAR10(root='./data', train=True, download=True)
        test_data = CIFAR10(root='./data', train=False, download=True)

        # Convert to numpy
        X_train = np.array(train_data.data)
        y_train = np.array(train_data.targets)
        X_test = np.array(test_data.data)
        y_test = np.array(test_data.targets)

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Run optimization
        aug = DeepAugment(X_train, y_train, X_test, y_test, train_size=2000, val_size=500)
        best = aug.optimize(iterations=iterations, epochs=epochs)

        print("\n✨ Best policy found:")
        for transform, magnitude in best:
            print(f"  {transform:20s} magnitude={magnitude:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure torchvision is installed: pip install torchvision")


def fashion_mnist(iterations=50, epochs=10):
    """Optimize augmentation for Fashion-MNIST dataset."""
    from .core import DeepAugment

    try:
        from torchvision.datasets import FashionMNIST

        print("Loading Fashion-MNIST...")
        train_data = FashionMNIST(root='./data', train=True, download=True)
        test_data = FashionMNIST(root='./data', train=False, download=True)

        # Convert to numpy and add channel dimension
        X_train = np.array(train_data.data.numpy())[..., None].repeat(3, axis=-1)
        y_train = np.array(train_data.targets.numpy())
        X_test = np.array(test_data.data.numpy())[..., None].repeat(3, axis=-1)
        y_test = np.array(test_data.targets.numpy())

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        aug = DeepAugment(X_train, y_train, X_test, y_test, train_size=2000, val_size=500)
        best = aug.optimize(iterations=iterations, epochs=epochs)

        print("\n✨ Best policy found:")
        for transform, magnitude in best:
            print(f"  {transform:20s} magnitude={magnitude:.2f}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """DeepAugment CLI - Find optimal image augmentation policies."""
    fire.Fire({
        'cifar10': cifar10,
        'fashion-mnist': fashion_mnist,
    })


if __name__ == '__main__':
    main()
