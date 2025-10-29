DeepAugment
===========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2949929.svg
   :target: https://doi.org/10.5281/zenodo.2949929
   :alt: DOI

.. image:: https://img.shields.io/pypi/v/deepaugment.svg?style=flat
   :target: https://pypi.org/project/deepaugment/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black

Find optimal image augmentation policies for your dataset automatically. DeepAugment uses Bayesian optimization to discover augmentation strategies that maximize model performance.

Resources: `blog post <https://medium.com/insight-data/automl-for-data-augmentation-e87cf692c366>`_, `slides <https://docs.google.com/presentation/d/1toRUTT9X26ACngr6DXCKmPravyqmaGjy-eIU5cTbG1A/edit#slide=id.g4cc092dbc6_0_0>`_

----

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install deepaugment

or using uv:

.. code-block:: bash

   uv add deepaugment

Simple API
~~~~~~~~~~

.. code-block:: python

   from deepaugment import optimize

   best_policy = optimize(my_images, my_labels, iterations=50)

Basic Usage (CIFAR-10)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchvision.datasets import CIFAR10
   from deepaugment import optimize
   import numpy as np

   train_data = CIFAR10(root='./data', train=True, download=True)
   X = np.array(train_data.data)[:5000]  # Use subset for speed
   y = np.array(train_data.targets)[:5000]

   best_policy = optimize(X, y, iterations=50)

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   from deepaugment import DeepAugment

   # Separate train/validation sets
   aug = DeepAugment(X_train, y_train, X_val, y_val,
                     n_operations=4,      # transforms per policy
                     train_size=2000,     # subset for speed
                     val_size=500)

   # Optimize
   best = aug.optimize(iterations=50, epochs=10)

   # Show results
   aug.show_best(n=5)

----

Why DeepAugment?
----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Fast & Efficient
      :text-align: center

      Uses Bayesian Optimization instead of Reinforcement Learning, requiring **~100x fewer iterations** than AutoAugment.

   .. grid-item-card:: 💰 Cost-Effective
      :text-align: center

      Optimize CIFAR-10 policies in **4.2 hours** for ~$13 on AWS p3.x2large (vs. days/weeks for AutoAugment).

   .. grid-item-card:: 🎯 Proven Results
      :text-align: center

      **60% error reduction** (8.5% accuracy increase) on CIFAR-10 with WRN-28-10.

   .. grid-item-card:: 🔧 Modular Design
      :text-align: center

      User-friendly API with extensive configuration options. Bring your own model or use built-in ones.

----

Key Features
------------

🎨 **26 Modern Transforms**
   Geometric, color, blur, noise, occlusion, and advanced transforms from torchvision v2.

🧠 **Smart Optimization**
   Bayesian Optimization with Random Forest Estimator and Expected Improvement acquisition.

⚡ **Minimal Child Model**
   Fast training (~30 seconds per iteration on V100 GPU) with 1.2M parameters.

🔬 **Scientific Rigor**
   Published method with DOI, reproducible results, and comprehensive documentation.

----

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/installation
   user-guide/basic-usage
   user-guide/advanced-usage
   user-guide/configuration

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/cifar10
   examples/custom-models

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: About

   about/how-it-works
   about/citation

----

How It Works
------------

DeepAugment has three major components:

1. **Controller**: Samples new augmentation policies using Bayesian Optimization
2. **Augmenter**: Transforms images according to the policy
3. **Child Model**: Trains from scratch on augmented images and returns a reward

.. image:: https://user-images.githubusercontent.com/14996155/52587711-797a4280-2def-11e9-84f8-2368fd709ab9.png
   :alt: DeepAugment workflow
   :align: center
   :width: 600px

The controller iteratively:
- Samples new policies based on previous results
- Evaluates them using the child model
- Updates its surrogate model
- Repeats until convergence or max iterations

----

Results
-------

CIFAR-10 best policies tested on WRN-28-10:

.. grid:: 2
   :gutter: 2

   .. grid-item::
      :columns: 6

      .. image:: https://user-images.githubusercontent.com/14996155/53362039-1d82e400-38ee-11e9-8f5e-e6f1602865a8.png
         :alt: Unaugmented results
         :width: 100%

   .. grid-item::
      :columns: 6

      .. image:: https://user-images.githubusercontent.com/14996155/53362042-21af0180-38ee-11e9-9253-96ce8ddcc17c.png
         :alt: DeepAugment results
         :width: 100%

**Result**: 60% reduction in error (8.5% accuracy increase) by DeepAugment

----

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
