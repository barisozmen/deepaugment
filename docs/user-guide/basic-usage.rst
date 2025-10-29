Basic Usage
===========

This guide covers the basic usage patterns of DeepAugment.

Quick Start
-----------

The simplest way to use DeepAugment is with the :func:`~deepaugment.optimize` function:

.. code-block:: python

   from deepaugment import optimize

   # Your dataset
   X_train = ...  # Shape: (N, H, W, C)
   y_train = ...  # Shape: (N,)

   # Find best augmentation policy
   best_policy = optimize(X_train, y_train, iterations=50)

The function will:

1. Automatically split your data into train/validation (80/20)
2. Run Bayesian optimization for 50 iterations
3. Display progress with a nice progress bar
4. Show top 5 best policies found
5. Return the best policy

Understanding Policies
----------------------

A policy is a list of augmentation operations with their magnitudes:

.. code-block:: python

   [
       ('rotate', 0.8),           # Rotate with 80% of max magnitude
       ('brightness', 0.5),       # Adjust brightness by 50%
       ('blur', 0.3),             # Apply blur at 30% strength
       ('flip_h', 0.9),           # Horizontal flip with 90% probability
   ]

Each operation is a tuple of ``(transform_name, magnitude)`` where magnitude is between 0.0 and 1.0.

CIFAR-10 Example
----------------

Complete example with CIFAR-10:

.. code-block:: python

   from torchvision.datasets import CIFAR10
   from deepaugment import optimize
   import numpy as np

   # Load CIFAR-10
   train_data = CIFAR10(root='./data', train=True, download=True)

   # Use a subset for speed (recommended for quick experiments)
   X = np.array(train_data.data)[:5000]
   y = np.array(train_data.targets)[:5000]

   # Optimize augmentation policy
   best_policy = optimize(X, y, iterations=50)

   # The policy is now ready to use for training your final model!

Custom Train/Validation Split
------------------------------

If you want control over the train/validation split:

.. code-block:: python

   from deepaugment import optimize

   # Your custom split
   X_train, y_train = ...
   X_val, y_val = ...

   best_policy = optimize(
       X_train, y_train,
       X_val, y_val,
       iterations=50
   )

Controlling Optimization
------------------------

Adjust the number of iterations and epochs:

.. code-block:: python

   from deepaugment import optimize

   best_policy = optimize(
       X_train, y_train,
       iterations=100,    # Try more policies (default: 50)
       epochs=15,         # Train longer per policy (default: 10)
       verbose=True       # Show progress bar (default: True)
   )

**Trade-offs:**

- **More iterations**: Better policies, but longer optimization time
- **More epochs**: More accurate evaluation, but slower per iteration
- **Recommended**: Start with 50 iterations and 10 epochs, then increase if needed

Using the DeepAugment Class
----------------------------

For more control, use the :class:`~deepaugment.DeepAugment` class directly:

.. code-block:: python

   from deepaugment import DeepAugment

   # Initialize
   aug = DeepAugment(X_train, y_train, X_val, y_val)

   # Optimize
   best_policy = aug.optimize(iterations=50, epochs=10)

   # View results
   aug.show_best(n=5)  # Show top 5 policies

   # Get best score
   print(f"Best validation accuracy: {aug.best_score():.3f}")

   # Access full history
   for entry in aug.history:
       print(f"Iteration {entry['iteration']}: {entry['score']:.3f}")

Reproducibility
---------------

Set a random seed for reproducible results:

.. code-block:: python

   from deepaugment import DeepAugment

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       random_state=42  # Fixed seed
   )

   best_policy = aug.optimize(iterations=50)

Every run with the same seed will produce the same results.

Device Selection
----------------

DeepAugment automatically uses the best available device (GPU if available):

.. code-block:: python

   from deepaugment import DeepAugment

   # Auto-detect (default)
   aug = DeepAugment(X_train, y_train, X_val, y_val, device="auto")

   # Force specific device
   aug = DeepAugment(X_train, y_train, X_val, y_val, device="cuda")
   aug = DeepAugment(X_train, y_train, X_val, y_val, device="mps")
   aug = DeepAugment(X_train, y_train, X_val, y_val, device="cpu")

Saving Results
--------------

DeepAugment can automatically save optimization history:

.. code-block:: python

   from deepaugment import DeepAugment

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       save_history=True,           # Enable auto-save
       experiment_name="my_exp"     # Custom name
   )

   best_policy = aug.optimize(iterations=50)

Results are saved in the ``experiments/`` directory as JSON files.

Next Steps
----------

- :doc:`advanced-usage` - Learn about advanced features
- :doc:`configuration` - Detailed configuration options
- :doc:`../examples/cifar10` - Complete CIFAR-10 example
- :doc:`../examples/custom-models` - Using custom models
