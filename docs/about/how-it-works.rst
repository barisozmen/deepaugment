How It Works
============

Technical deep-dive into DeepAugment's design and methodology.

Overview
--------

DeepAugment automates the search for optimal image augmentation policies using Bayesian Optimization. It consists of three main components that work together in an iterative loop:

1. **Controller**: Samples augmentation policies using Bayesian Optimization
2. **Augmenter**: Transforms images according to policies
3. **Child Model**: Evaluates policy quality through training

.. image:: https://user-images.githubusercontent.com/14996155/52587711-797a4280-2def-11e9-84f8-2368fd709ab9.png
   :alt: DeepAugment workflow
   :align: center
   :width: 600px

----

The Optimization Loop
---------------------

The core workflow:

.. code-block:: text

   1. Controller samples a new augmentation policy
   2. Augmenter applies the policy to training images
   3. Child model trains on augmented images
   4. Validation accuracy is computed (reward)
   5. Controller updates with (policy, reward) pair
   6. Repeat until convergence or max iterations

This process discovers which augmentation combinations work best for your specific dataset.

----

Why Bayesian Optimization?
---------------------------

Comparison of Hyperparameter Optimization Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Iterations Needed
     - Computation Cost
     - Accuracy
     - Complexity
   * - Grid Search
     - Very High
     - Very High
     - Medium
     - Low
   * - Random Search
     - High
     - High
     - Medium
     - Low
   * - Bayesian Optimization
     - Low (~100-300)
     - Low
     - High
     - Medium
   * - Reinforcement Learning
     - Very High (~15,000)
     - Very High
     - High
     - High

.. image:: https://user-images.githubusercontent.com/14996155/53222123-4ae73d80-3621-11e9-9457-44e76012d11c.png
   :alt: Optimization comparison
   :align: center
   :width: 500px

AutoAugment vs DeepAugment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Google's AutoAugment uses Reinforcement Learning:

- **Iterations needed**: ~15,000
- **Time**: Days to weeks
- **Cost**: Requires massive computational resources
- **Accessibility**: Not practical for most users

DeepAugment uses Bayesian Optimization:

- **Iterations needed**: ~100-300
- **Time**: Hours
- **Cost**: ~$13 on AWS for CIFAR-10
- **Accessibility**: Practical for individual researchers and small teams

**Performance**: Bayesian Optimization achieves comparable or better results with **~100x fewer iterations**.

----

Bayesian Optimization Details
------------------------------

How It Works
~~~~~~~~~~~~

Bayesian Optimization maintains a **surrogate model** that predicts the quality of unexplored policies:

1. **Build surrogate model** from previous evaluations
2. **Acquisition function** identifies promising policies to try next
3. **Evaluate** the selected policy
4. **Update** surrogate model with new result
5. **Repeat**

DeepAugment uses:

- **Surrogate**: Random Forest Estimator
- **Acquisition**: Expected Improvement (EI)
- **Library**: scikit-optimize

Expected Improvement
~~~~~~~~~~~~~~~~~~~~

The acquisition function balances:

- **Exploitation**: Try policies similar to current best
- **Exploration**: Try unexplored regions of policy space

This balance is key to efficient optimization.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

The optimization problem:

.. math::

   \\mathbf{p}^* = \\arg\\max_{\\mathbf{p} \\in \\mathcal{P}} f(\\mathbf{p})

Where:

- :math:`\\mathbf{p}` is an augmentation policy
- :math:`\\mathcal{P}` is the space of all possible policies
- :math:`f(\\mathbf{p})` is the validation accuracy with policy :math:`\\mathbf{p}`
- :math:`\\mathbf{p}^*` is the optimal policy

The challenge: :math:`f(\\mathbf{p})` is expensive to evaluate (requires training a model).

Bayesian Optimization efficiently explores :math:`\\mathcal{P}` by building a probabilistic model of :math:`f`.

----

Policy Representation
---------------------

Policy Structure
~~~~~~~~~~~~~~~~

A policy consists of :math:`N` operations (default :math:`N=4`):

.. math::

   \\mathbf{p} = [(t_1, m_1), (t_2, m_2), ..., (t_N, m_N)]

Where:

- :math:`t_i` is a transform type (categorical: 1 to 26)
- :math:`m_i` is magnitude (continuous: 0.0 to 1.0)

Example policy:

.. code-block:: python

   [
       ('rotate', 0.8),      # t₁=rotate, m₁=0.8
       ('brightness', 0.5),  # t₂=brightness, m₂=0.5
       ('blur', 0.3),        # t₃=blur, m₃=0.3
       ('flip_h', 0.9),      # t₄=flip_h, m₄=0.9
   ]

Search Space Size
~~~~~~~~~~~~~~~~~

For :math:`N=4` operations with 26 transforms:

- Categorical dimensions: 26 choices × 4 = :math:`26^4 = 456,976` combinations
- Continuous dimensions: :math:`[0, 1]^4` (infinite)
- **Total**: Extremely large search space

This is why naive grid search is infeasible and Bayesian Optimization is necessary.

----

Transform Library
-----------------

DeepAugment includes 26 modern transforms from torchvision v2:

Geometric Transforms (8)
~~~~~~~~~~~~~~~~~~~~~~~~

- ``rotate``: Rotation by angle
- ``flip_h``: Horizontal flip
- ``flip_v``: Vertical flip
- ``affine``: Affine transformation
- ``shear``: Shear transformation
- ``perspective``: Perspective transformation
- ``elastic``: Elastic deformation
- ``random_crop``: Random cropping

Color Transforms (5)
~~~~~~~~~~~~~~~~~~~~

- ``brightness``: Brightness adjustment
- ``contrast``: Contrast adjustment
- ``saturation``: Saturation adjustment
- ``hue``: Hue adjustment
- ``color_jitter``: Combined color jittering

Advanced Color (7)
~~~~~~~~~~~~~~~~~~

- ``sharpen``: Sharpening
- ``autocontrast``: Auto contrast
- ``equalize``: Histogram equalization
- ``invert``: Color inversion
- ``solarize``: Solarization
- ``posterize``: Posterization
- ``grayscale``: Grayscale conversion

Blur & Noise (2)
~~~~~~~~~~~~~~~~

- ``blur``: Gaussian blur
- ``gaussian_noise``: Additive Gaussian noise

Occlusion (2)
~~~~~~~~~~~~~

- ``erasing``: Random erasing
- ``cutout``: Cutout augmentation

Advanced (2)
~~~~~~~~~~~~

- ``channel_permute``: Channel permutation
- ``photometric_distort``: Photometric distortion

Each transform's magnitude is normalized to [0, 1] for uniform optimization.

----

Child Model
-----------

Architecture
~~~~~~~~~~~~

The child model is a lightweight CNN designed for fast training:

- **Parameters**: 1,250,858 (for 32×32 images)
- **Training time**: ~30 seconds per iteration on V100 GPU
- **Architecture**: 3 convolutional blocks + fully connected layers

.. image:: https://user-images.githubusercontent.com/14996155/52545277-10e98200-2d6b-11e9-9639-48b671711eba.png
   :alt: Child CNN architecture
   :align: center
   :width: 800px

Design Principles
~~~~~~~~~~~~~~~~~

The child model is intentionally small:

1. **Fast evaluation**: Each policy needs training from scratch
2. **Good proxy**: Performance correlates with larger models
3. **Memory efficient**: Fits in GPU memory with large batches

**Key insight**: Small model + good augmentation ≈ Large model + weak augmentation

Custom Models
~~~~~~~~~~~~~

You can use your own model as the child model:

.. code-block:: python

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       model=MyCustomModel
   )

Trade-off: Larger models give more accurate policy evaluation but take longer.

----

Reward Function
---------------

Default Reward
~~~~~~~~~~~~~~

The reward is the validation accuracy of the child model trained with the policy:

.. math::

   r(\\mathbf{p}) = \\text{Accuracy}_{\\text{val}}(\\text{Model trained with } \\mathbf{p})

Implementation details:

- Model trained for :math:`E` epochs (default :math:`E=10`)
- Reward is mean of top :math:`K` validation accuracies (default :math:`K=3`)
- This reduces noise from training variance

Custom Rewards
~~~~~~~~~~~~~~

You can define custom reward functions:

.. code-block:: python

   def my_reward(entry):
       score = entry['score']
       policy = entry['policy']

       # Example: Penalize complex policies
       complexity = len(policy)
       return score - 0.01 * complexity

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       custom_reward_fn=my_reward
   )

This allows optimizing for multiple objectives (accuracy + simplicity, speed, etc.).

----

Data Pipeline
-------------

Training Data Flow
~~~~~~~~~~~~~~~~~~

.. image:: https://user-images.githubusercontent.com/14996155/52740938-0d334680-2f89-11e9-8d68-117d139d9ab8.png
   :alt: Data pipeline training
   :align: center
   :width: 600px

Validation Data Flow
~~~~~~~~~~~~~~~~~~~~

.. image:: https://user-images.githubusercontent.com/14996155/52740937-0c9ab000-2f89-11e9-9e94-beca71caed41.png
   :alt: Data pipeline validation
   :align: center
   :width: 600px

Key Points
~~~~~~~~~~

1. **Augmentation applied per-epoch**: Same image gets different augmentations each epoch
2. **Validation not augmented**: Ensures unbiased evaluation
3. **Random sampling**: Magnitude determines probability/intensity of each transform
4. **Sequential application**: Transforms applied in policy order

----

Design Principles
-----------------

DeepAugment follows several design philosophies:

Convention over Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sensible defaults for everything:

.. code-block:: python

   # This works out of the box
   best = optimize(X, y, iterations=50)

Rails Doctrine
~~~~~~~~~~~~~~

1. **Optimize for programmer happiness**: Clean API, readable code
2. **Convention over configuration**: Defaults work well
3. **Progress over stability**: Use modern approaches
4. **Omakase**: Curated, opinionated stack

Single Source of Truth
~~~~~~~~~~~~~~~~~~~~~~~

Each piece of logic lives in exactly one place:

- Policy representation → ``policy.py``
- Transforms → ``transforms.py``
- Training → ``trainer.py``
- Search → ``search.py``

This makes the codebase maintainable and extensible.

----

Academic Foundation
-------------------

DeepAugment builds on strong theoretical foundations:

Key Papers
~~~~~~~~~~

1. **AutoAugment** (Cubuk et al., 2018): Original idea of learned augmentation
2. **Bayesian Optimization Review** (Shahriari et al., 2016): BO theory
3. **Neural Architecture Search** (Zoph et al., 2016): Search methodology
4. **Cutout** (DeVries & Taylor, 2017): Occlusion augmentation

Novel Contributions
~~~~~~~~~~~~~~~~~~~

DeepAugment's contributions:

1. **First application of Bayesian Optimization** to augmentation policy search
2. **Minimized child model** for computational efficiency
3. **Practical implementation** accessible to individual researchers
4. **Open source** with complete code and documentation

Performance Validation
~~~~~~~~~~~~~~~~~~~~~~

Validated on CIFAR-10 with WRN-28-10:

- **Baseline**: 91.5% accuracy
- **With DeepAugment**: 95.0% accuracy
- **Improvement**: 8.5% absolute (60% error reduction)

See :doc:`citation` for how to cite this work.

----

Computational Complexity
------------------------

Time Complexity
~~~~~~~~~~~~~~~

For :math:`T` iterations, :math:`E` epochs, :math:`N` samples, batch size :math:`B`:

.. math::

   \\text{Time} \\approx T \\times E \\times \\frac{N}{B} \\times t_{\\text{forward+backward}}

For CIFAR-10 with default settings:

- :math:`T=100` iterations
- :math:`E=10` epochs
- :math:`N=2000` samples
- :math:`B=64` batch size
- :math:`t=0.01`s per batch on V100

Total time: ~4.2 hours (~$13 on AWS p3.x2large)

Space Complexity
~~~~~~~~~~~~~~~~

Memory usage:

- Model parameters: ~1.2M × 4 bytes = 5 MB
- Batch storage: batch_size × image_size × 4 bytes
- Optimizer state: 2× model size = 10 MB

Total: ~100-200 MB on GPU (very efficient)

----

See Also
--------

- :doc:`citation` - How to cite DeepAugment
- :doc:`../user-guide/advanced-usage` - Advanced usage patterns
- :doc:`../api/index` - API documentation

References
~~~~~~~~~~

- `AutoAugment paper <https://arxiv.org/abs/1805.09501>`_
- `Bayesian Optimization review <https://ieeexplore.ieee.org/document/7352306>`_
- `Blog post on Bayesian Optimization <https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f>`_
