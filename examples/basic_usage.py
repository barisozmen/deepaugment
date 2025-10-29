"""Basic usage example of DeepAugment."""

import numpy as np
from deepaugment import DeepAugment, optimize

# Example 1: Simple usage with optimize() function
def example_simple():
    """Quick optimization with built-in datasets."""
    from torchvision.datasets import CIFAR10

    print('Loading CIFAR-10 data...')
    train_data = CIFAR10(root='./data', train=True, download=True)
    X = np.array(train_data.data)[:5000]  # Use subset for speed
    y = np.array(train_data.targets)[:5000]

    print('Finding the best policy...')
    # automatically splits train/val)
    best_policy = optimize(X, y, iterations=30)

    print("Best policy:", best_policy)


# Example 2: Full control with DeepAugment class
def example_full_control():
    """Full control over optimization process."""
    from torchvision.datasets import CIFAR10

    # Load data
    train_data = CIFAR10(root='./data', train=True, download=True)
    test_data = CIFAR10(root='./data', train=False, download=True)

    X_train = np.array(train_data.data)
    y_train = np.array(train_data.targets)
    X_test = np.array(test_data.data)
    y_test = np.array(test_data.targets)

    # Initialize optimizer
    aug = DeepAugment(
        X_train, y_train, X_test, y_test,
        n_operations=4,  # Number of transforms per policy
        train_size=2000,  # Subset for faster iteration
        val_size=500,
    )

    # Run optimization
    best = aug.optimize(iterations=50, epochs=10, samples=1)

    # Show results
    aug.show_best(n=3)

    return best


# Example 3: Use discovered policy
def example_apply_policy():
    """Apply discovered policy to augment images."""
    from deepaugment import apply_policy
    import matplotlib.pyplot as plt
    from torchvision.datasets import CIFAR10

    # Load sample image
    data = CIFAR10(root='./data', train=True, download=True)
    image = np.array(data.data[np.random.randint(len(data))])

    # Define a policy (transform_name, magnitude)
    policy = [
        ('rotate', 0.3),
        ('blur', 0.2),
        ('brightness', 0.4),
        ('flip_h', 0.5),
    ]

    # Apply policy
    augmented = apply_policy(image, policy)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[1].imshow(augmented)
    axes[1].set_title('Augmented')
    plt.show()


if __name__ == '__main__':
    
    while True:
        example = input('Which example?\n  1-simple\n  2-full_control\n  3-apply_policy\n')
        if example in ['1', '2', '3']:
            break
        else:
            print('Invalid example. Please select again\n\n')
    
    match example:
        case '1':
            example_simple()
        case '2':
            example_full_control()
        case '3':
            example_apply_policy()
