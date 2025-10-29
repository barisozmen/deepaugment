Configuration
=============

Complete reference for all configuration options in DeepAugment.

DeepAugment Initialization
---------------------------

.. code-block:: python

   from deepaugment import DeepAugment

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
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
       # Core parameters
       n_operations=4,
       train_size=2000,
       val_size=500,
   )

Essential Parameters
~~~~~~~~~~~~~~~~~~~~

**model** : ``str`` or ``nn.Module`` class, default="simple"
    Model architecture to use for evaluation.

    - ``"simple"``: SimpleCNN (default, 1.2M parameters, fast)
    - Custom PyTorch model class (see :doc:`advanced-usage`)

**device** : ``str``, default="auto"
    Device for training.

    - ``"auto"``: Automatically select best device (CUDA > MPS > CPU)
    - ``"cuda"``: Force NVIDIA GPU
    - ``"mps"``: Force Apple Silicon GPU
    - ``"cpu"``: Force CPU

**random_state** : ``int`` or ``None``, default=None
    Random seed for reproducibility. If None, uses default from config (42).

**experiment_name** : ``str`` or ``None``, default=None
    Name for the experiment. If None, generates timestamp-based name.

Useful Parameters
~~~~~~~~~~~~~~~~~

**method** : ``str``, default="bayesian"
    Search strategy.

    - ``"bayesian"``: Bayesian Optimization (recommended)
    - ``"random"``: Random search (baseline)

**save_history** : ``bool``, default=True
    Whether to save optimization history to disk. Saved in ``experiments/`` directory.

Advanced Parameters
~~~~~~~~~~~~~~~~~~~

**transform_categories** : ``list`` or ``None``, default=None
    Filter transforms by category. If None, uses all transforms.

    Available categories:

    - ``"geometric"``: rotate, flip, affine, shear, perspective, elastic, random_crop
    - ``"color"``: brightness, contrast, saturation, hue, color_jitter
    - ``"advanced_color"``: sharpen, autocontrast, equalize, invert, solarize, posterize, grayscale
    - ``"blur_noise"``: blur, gaussian_noise
    - ``"occlusion"``: erasing, cutout
    - ``"advanced"``: channel_permute, photometric_distort

    Example:

    .. code-block:: python

       transform_categories=["geometric", "color"]

**custom_reward_fn** : ``callable`` or ``None``, default=None
    Custom reward function for optimization. Function signature:

    .. code-block:: python

       def my_reward(entry: dict) -> float:
           """
           Args:
               entry: Dict with keys:
                   - 'policy': List of (transform, magnitude) tuples
                   - 'score': Validation accuracy
                   - 'iteration': Current iteration number

           Returns:
               Custom reward value
           """
           return entry['score']  # Default behavior

**resume_from** : ``str`` or ``None``, default=None
    Path to checkpoint file to resume optimization from.

Core Parameters
~~~~~~~~~~~~~~~

**n_operations** : ``int``, default=4
    Number of augmentation operations per policy. Higher values create more complex policies but increase search space.

**train_size** : ``int``, default=2000
    Number of training samples to use for optimization. Using a subset speeds up optimization.

**val_size** : ``int``, default=500
    Number of validation samples to use for evaluation.

----

Optimize Method
---------------

.. code-block:: python

   best_policy = aug.optimize(
       iterations=50,
       epochs=10,
       samples=1,
       batch_size=64,
       learning_rate=0.001,
       early_stopping=False,
       patience=10,
       verbose=True,
   )

Parameters
~~~~~~~~~~

**iterations** : ``int``, default=50
    Number of policies to try. More iterations generally lead to better policies but take longer.

    - Quick experiments: 25-50
    - Standard: 50-100
    - Thorough: 100-300

**epochs** : ``int``, default=10
    Training epochs per policy evaluation. More epochs give more accurate evaluation but are slower.

    - Quick: 5-10
    - Standard: 10-15
    - Thorough: 15-20

**samples** : ``int``, default=1
    Number of training runs per policy. Results are averaged to reduce noise.

    - Fast: 1
    - Stable: 3
    - Very stable: 5

**batch_size** : ``int``, default=64
    Training batch size.

**learning_rate** : ``float``, default=0.001
    Learning rate for training.

**early_stopping** : ``bool``, default=False
    Stop optimization if no improvement is seen.

**patience** : ``int``, default=10
    Number of iterations without improvement before stopping (only if ``early_stopping=True``).

**verbose** : ``bool``, default=True
    Show progress bar and updates.

----

Default Configuration
---------------------

DeepAugment uses sensible defaults from ``deepaugment.config.defaults``:

.. code-block:: python

   from deepaugment.config import defaults

   print(f"Default iterations: {defaults.iterations}")
   print(f"Default epochs: {defaults.epochs}")
   print(f"Default n_operations: {defaults.n_operations}")

   # View all defaults
   print(defaults)

These defaults are optimized for:

- Balance between speed and quality
- Quick iteration during development
- Reasonable computational cost

You can override any default by passing it explicitly to ``DeepAugment()`` or ``optimize()``.

----

Environment Variables
---------------------

**DEEPAUGMENT_EXPERIMENTS_DIR**
    Override default experiments directory. Default: ``./experiments``

    .. code-block:: bash

       export DEEPAUGMENT_EXPERIMENTS_DIR=/path/to/experiments

**DEEPAUGMENT_DEVICE**
    Override default device selection. Values: ``cuda``, ``mps``, ``cpu``

    .. code-block:: bash

       export DEEPAUGMENT_DEVICE=cuda

----

Examples
--------

Quick Experiment
~~~~~~~~~~~~~~~~

.. code-block:: python

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       train_size=500,
       val_size=100,
   )
   best = aug.optimize(iterations=25, epochs=5)

Thorough Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       train_size=5000,
       val_size=1000,
       save_history=True,
       experiment_name="thorough_run"
   )
   best = aug.optimize(
       iterations=200,
       epochs=20,
       samples=3,
       early_stopping=True,
       patience=20
   )

Reproducible Run
~~~~~~~~~~~~~~~~

.. code-block:: python

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       random_state=42,
       device="cuda",
   )
   best = aug.optimize(iterations=50, epochs=10)
   # Same results every time with same seed

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       model=MyCustomModel,
       transform_categories=["geometric", "color"],
       custom_reward_fn=my_reward,
       n_operations=6,
       method="bayesian",
       save_history=True,
   )
   best = aug.optimize(
       iterations=100,
       epochs=15,
       samples=2,
       batch_size=128,
       learning_rate=0.002,
   )

----

See Also
--------

- :doc:`basic-usage` - Basic usage patterns
- :doc:`advanced-usage` - Advanced features and customization
- :doc:`../api/index` - Complete API reference
