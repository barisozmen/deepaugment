# DeepAugment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2949929.svg)](https://doi.org/10.5281/zenodo.2949929)

![pypi](https://img.shields.io/pypi/v/deepaugment.svg?style=flat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

DeepAugment discovers augmentation strategies tailored for your images. It uses Bayesian Optimization for optimizing data augmentation hyperparameters.

## Installation/Usage

1.  **Install uv:** Follow the instructions at https://astral.sh/uv/install.sh
2.  **Initialize the project and install dependencies:**
    ```console
    uv init
    uv sync
    ```

### Simple usage (with any dataset)
```Python
from deepaugment.deepaugment import DeepAugment

deepaug = DeepAugment(my_images, my_labels)

best_policies = deepaug.optimize(300)
```

### Simple usage (with CIFAR-10 on keras)
```Python
deepaug = DeepAugment("cifar10")

best_policies = deepaug.optimize(300)
```

### Advanced usage
```Python
from tensorflow.keras.datasets import fashion_mnist

# my configuration
my_config = {
    "model": "basiccnn",
    "method": "bayesian_optimization",
    "train_set_size": 2000,
    "opt_samples": 3,
    "opt_last_n_epochs": 3,
    "opt_initial_points": 10,
    "child_epochs": 50,
    "child_first_train_epochs": 0,
    "child_batch_size": 64
}

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
deepaug = DeepAugment(images=x_train, labels=y_train, config=my_config)

best_policies = deepaug.optimize(300)
```

## How it works

DeepAugment uses Bayesian Optimization to find the best image augmentation policies for a given dataset. It works by iteratively training a small "child" model with different augmentation policies and uses the model's performance as a reward signal to guide the search for better policies.

## References
[1] Cubuk et al., 2018. AutoAugment: Learning Augmentation Policies from Data
([arxiv](https://arxiv.org/abs/1805.09501))

[2] Shahriari et al., 2016. A review of Bayesian Optimization
([ieee](https://ieeexplore.ieee.org/document/7352306))

--------

## Contact
Baris Ozmen, hbaristr@gmail.com
Peter Norvig
