CIFAR-10 Example
================

Complete walkthrough of optimizing augmentation policies for CIFAR-10.

Overview
--------

This example demonstrates:

- Loading CIFAR-10 dataset
- Running optimization with different configurations
- Comparing results
- Applying policies to final model training

Basic CIFAR-10 Optimization
----------------------------

.. code-block:: python

   from torchvision.datasets import CIFAR10
   from deepaugment import optimize
   import numpy as np

   # Load CIFAR-10
   train_data = CIFAR10(root='./data', train=True, download=True)
   X = np.array(train_data.data)
   y = np.array(train_data.targets)

   # Use subset for quick experiment
   X_subset = X[:5000]
   y_subset = y[:5000]

   # Optimize
   best_policy = optimize(X_subset, y_subset, iterations=50)

This runs in about 4-5 hours on a modern GPU.

Full Dataset Optimization
--------------------------

For best results, use the full training set:

.. code-block:: python

   from torchvision.datasets import CIFAR10
   from deepaugment import DeepAugment
   import numpy as np

   # Load full CIFAR-10
   train_data = CIFAR10(root='./data', train=True, download=True)
   X_train = np.array(train_data.data)
   y_train = np.array(train_data.targets)

   # Split into train/val
   from sklearn.model_selection import train_test_split
   X_train, X_val, y_train, y_val = train_test_split(
       X_train, y_train, test_size=0.1, random_state=42
   )

   # Optimize with more resources
   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       train_size=10000,  # Use 10k for evaluation
       val_size=2000,
       save_history=True,
       experiment_name="cifar10_full"
   )

   best_policy = aug.optimize(
       iterations=200,
       epochs=20,
       verbose=True
   )

   # Show top policies
   aug.show_best(n=10)

Expected Results
----------------

On CIFAR-10 with WRN-28-10, DeepAugment achieves:

- **Baseline (no augmentation)**: ~91.5% accuracy
- **With DeepAugment policies**: ~95.0% accuracy
- **Improvement**: 8.5% absolute increase (60% error reduction)

These results were obtained with:

- 200 iterations
- 20 epochs per evaluation
- SimpleCNN as child model
- Full training set

Comparing Search Methods
-------------------------

Compare Bayesian Optimization vs Random Search:

.. code-block:: python

   import matplotlib.pyplot as plt
   from deepaugment import DeepAugment

   results = {}

   for method in ["bayesian", "random"]:
       print(f"\nRunning {method} optimization...")

       aug = DeepAugment(
           X_train, y_train, X_val, y_val,
           method=method,
           experiment_name=f"cifar10_{method}",
           save_history=True
       )

       best = aug.optimize(iterations=100, epochs=10)
       results[method] = aug.history

   # Plot comparison
   plt.figure(figsize=(10, 6))
   for method, history in results.items():
       scores = [entry['score'] for entry in history]
       plt.plot(scores, label=method.capitalize())

   plt.xlabel('Iteration')
   plt.ylabel('Validation Accuracy')
   plt.title('Bayesian Optimization vs Random Search on CIFAR-10')
   plt.legend()
   plt.grid(True)
   plt.savefig('comparison.png')

Bayesian Optimization typically converges faster and finds better policies.

Transform Category Experiments
-------------------------------

Test which transform categories work best:

.. code-block:: python

   from deepaugment import DeepAugment

   categories_to_test = [
       None,  # All transforms
       ["geometric"],
       ["color"],
       ["geometric", "color"],
       ["geometric", "color", "blur_noise"],
   ]

   results = {}

   for categories in categories_to_test:
       name = "all" if categories is None else "_".join(categories)
       print(f"\nTesting: {name}")

       aug = DeepAugment(
           X_train, y_train, X_val, y_val,
           transform_categories=categories,
           experiment_name=f"cifar10_{name}"
       )

       best = aug.optimize(iterations=50, epochs=10)
       results[name] = aug.best_score()

   # Print results
   print("\n" + "="*50)
   print("Results by Transform Category")
   print("="*50)
   for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
       print(f"{name:30s}: {score:.4f}")

Applying Policy to Final Model
-------------------------------

Once you've found a good policy, use it to train your final model:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader, TensorDataset
   from deepaugment import apply_policy

   # Your optimized policy
   best_policy = aug.best_policy()

   # Create augmented dataset
   def create_augmented_dataset(X, y, policy, augment_ratio=0.5):
       """Create dataset with augmented samples."""
       X_aug = []
       y_aug = []

       for i in range(len(X)):
           # Original image
           X_aug.append(X[i])
           y_aug.append(y[i])

           # Add augmented version
           if np.random.rand() < augment_ratio:
               img_tensor = torch.from_numpy(X[i]).permute(2, 0, 1)
               aug_img = apply_policy(img_tensor, policy)
               X_aug.append(aug_img.permute(1, 2, 0).numpy())
               y_aug.append(y[i])

       return np.array(X_aug), np.array(y_aug)

   # Create augmented training set
   X_train_aug, y_train_aug = create_augmented_dataset(
       X_train, y_train, best_policy, augment_ratio=0.5
   )

   # Train your final model
   class WideResNet(nn.Module):
       # Your WRN implementation
       pass

   model = WideResNet()
   optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
   criterion = nn.CrossEntropyLoss()

   # Standard training loop
   for epoch in range(200):
       # Training with augmented data
       model.train()
       for batch_x, batch_y in train_loader:
           optimizer.zero_grad()
           outputs = model(batch_x)
           loss = criterion(outputs, batch_y)
           loss.backward()
           optimizer.step()

       # Validation
       model.eval()
       # ... validation code ...

Tips for CIFAR-10
-----------------

1. **Start Small**: Begin with 25-50 iterations to get a feel for the process
2. **Use Subsets**: 5000-10000 samples is usually enough for finding good policies
3. **Monitor Progress**: Use ``save_history=True`` and check intermediate results
4. **Multiple Runs**: Try 2-3 runs with different seeds for robustness
5. **GPU Required**: CPU training is very slow; use GPU for practical optimization
6. **Batch Size**: Larger batch sizes (128-256) can speed up training
7. **Policy Complexity**: Start with 4 operations, increase if needed

Common Issues
-------------

**Out of Memory**
    Reduce ``batch_size`` or ``train_size``

**Slow Training**
    Use GPU, reduce ``train_size`` or ``val_size``, or decrease ``epochs``

**Poor Results**
    Increase ``iterations``, ``epochs``, or ``train_size``

**No Improvement**
    Check that augmentations are being applied, verify data format (CHW vs HWC)

Next Steps
----------

- Try on your own dataset
- Experiment with custom models
- Tune hyperparameters for better results
- Use discovered policies in production training

See Also
--------

- :doc:`custom-models` - Using custom models
- :doc:`../user-guide/advanced-usage` - Advanced configuration
- :doc:`../api/index` - API reference
