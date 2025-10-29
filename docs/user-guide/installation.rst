Installation
============

DeepAugment can be installed via pip or uv (recommended for faster installs).

Requirements
------------

- Python >= 3.11
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended for faster training)

Install from PyPI
-----------------

Using pip:

.. code-block:: bash

   pip install deepaugment

Using uv (faster):

.. code-block:: bash

   uv add deepaugment

Install from Source
-------------------

For development or the latest features:

.. code-block:: bash

   git clone https://github.com/barisozmen/deepaugment.git
   cd deepaugment
   uv sync
   uv pip install -e .

Verify Installation
-------------------

Check that DeepAugment is properly installed:

.. code-block:: python

   import deepaugment
   print(deepaugment.__version__)

You can also check available transforms:

.. code-block:: python

   from deepaugment import TRANSFORMS
   print(f"Available transforms: {len(TRANSFORMS)}")

Dependencies
------------

DeepAugment requires the following packages:

- **numpy** >= 1.24.0 - Numerical operations
- **torch** >= 2.0.0 - Deep learning framework
- **torchvision** >= 0.15.0 - Image transformations
- **scikit-optimize** >= 0.10.2 - Bayesian optimization
- **tqdm** >= 4.66.0 - Progress bars
- **matplotlib** >= 3.10.7 - Visualization
- **fire** >= 0.6.0 - CLI interface
- **attrs** >= 25.4.0 - Clean class definitions
- **cattrs** >= 25.3.0 - Serialization

All dependencies are automatically installed with the package.

GPU Support
-----------

DeepAugment automatically detects and uses available GPUs. To verify GPU support:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"MPS available: {torch.backends.mps.is_available()}")

Device Selection
~~~~~~~~~~~~~~~~

DeepAugment automatically selects the best available device by default:

.. code-block:: python

   from deepaugment import DeepAugment

   # Auto-detect best device (default)
   aug = DeepAugment(X_train, y_train, X_val, y_val, device="auto")

   # Force specific device
   aug = DeepAugment(X_train, y_train, X_val, y_val, device="cuda")  # NVIDIA GPU
   aug = DeepAugment(X_train, y_train, X_val, y_val, device="mps")   # Apple Silicon
   aug = DeepAugment(X_train, y_train, X_val, y_val, device="cpu")   # CPU only

Next Steps
----------

- :doc:`basic-usage` - Learn basic usage patterns
- :doc:`advanced-usage` - Explore advanced features
- :doc:`configuration` - Configure DeepAugment for your needs
