# DeepAugment

[![GitHub last commit](https://img.shields.io/github/last-commit/barisozmen/deepaugment.svg)](https://github.com/barisozmen/deepaugment/commits/master) [![Downloads](https://static.pepy.tech/badge/deepaugment)](https://pepy.tech/project/deepaugment) ![pypi](https://img.shields.io/pypi/v/deepaugment.svg?style=flat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/deepaugment/badge/?version=latest)](https://deepaugment.readthedocs.io/en/latest/?badge=latest)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2949929.svg)](https://doi.org/10.5281/zenodo.2949929)



Find optimal image augmentation policies for your dataset automatically. DeepAugment uses Bayesian optimization to discover augmentation strategies that maximize model performance.

Resources: [blog post](https://medium.com/insight-data/automl-for-data-augmentation-e87cf692c366), [slides](https://docs.google.com/presentation/d/1toRUTT9X26ACngr6DXCKmPravyqmaGjy-eIU5cTbG1A/edit#slide=id.g4cc092dbc6_0_0)

## Quick Start

```bash
$ pip install deepaugment # (or `$ uv add deepaugment`)
```

### Simple API

```python
from deepaugment import optimize

best_policy = optimize(my_images, my_labels, iterations=50)
```

### Simple usage (CIFAR-10 example)

```python
from torchvision.datasets import CIFAR10
from deepaugment import optimize

train_data = CIFAR10(root='./data', train=True, download=True)
X = np.array(train_data.data)[:5000]
y = np.array(train_data.targets)[:5000]

best_policy = optimize(X, y, iterations=50)
```

### Advanced usage

```python
from torchvision.datasets import CIFAR10
from deepaugment import DeepAugment

aug = DeepAugment(
    # Data
    X_train, y_train,
    X_val, y_val,
    
    # Parameters
    n_operations=4,      # transforms per policy
    train_size=2000,     
    val_size=500
)

# Optimize
best = aug.optimize(iterations=50, epochs=10)

# Show results
aug.show_best(n=5)
```
## Results
### CIFAR-10 best policies tested on WRN-28-10
- Method: Wide-ResNet-28-10 trained with CIFAR-10 augmented images by best found policies, and with unaugmented images (everything else same).
- Result: **60% reduction in error** (8.5% accuracy increase) by DeepAugment
<img src="https://user-images.githubusercontent.com/14996155/53362039-1d82e400-38ee-11e9-8f5e-e6f1602865a8.png" width="400"> <img src="https://user-images.githubusercontent.com/14996155/53362042-21af0180-38ee-11e9-9253-96ce8ddcc17c.png" width="400">

## Design goals
DeepAugment is designed as a scalable and modular partner to AutoAugment ([Cubuk et al., 2018](https://arxiv.org/abs/1805.09501)). AutoAugment was one of the most exciting publications in 2018. It was the first method using Reinforcement Learning for this problem. AutoAugmentation, however, has no complete open-sourced implementation (controller module not available) preventing users to run it for their own datasets, and takes 15,000 iterations to learn (according to paper) augmentation policies, which requires massive computational resources. Thus most people could not benefit from it even if its source code would be fully available.

DeepAugment addresses these two problems. Its main design goals are:
1. **minimize the computational complexity of optimization while maintaining quality of results**
2. **be modular and user-friendly**

First goal is achieved by following changes compared to AutoAugment:
1. Bayesian Optimization instead of Reinforcement Learning
    * which requires much less number of iterations (~100 times)
2. Minimized Child Model
    * decreasing computational complexity of each training (~20 times)
3. Less stochastic augmentation search space design
    * decreasing number of iterations needed

For achieving the second goal, user interface is designed in a way that it gives user broad configuration possibilities and model selections (e.g. selecting the child model or inputting a self-designed child model).

## Importance
### Practical importance
DeepAugment makes optimization of data augmentation scalable, and thus enables users to optimize augmentation policies without needing massive computational resources.
As an estimate of its computational cost, it takes **4.2 hours** (500 iterations) on CIFAR-10 dataset which costs around **$13** using AWS p3.x2large instance.
### Academic importance
To our knowledge, DeepAugment is the first method which utilizes Bayesian Optimization for the problem of data augmentation hyperparameter optimization.

## How it works

Three major components of DeepAugment are controller, augmenter, and child model. Overall workflow is that controller samples new augmentation policies, augmenter transforms images by the new policy, and child model is trained from scratch by augmented images. Then, a reward is calculated from child model's training history. This reward is returned back to the controller, and it updates its surrogate model with this reward and associated augmentation policy. Then, controller samples new policies again and same steps repeats. This process cycles until user-determined maximum number of iterations reached.

Controller can be set for using either Bayesian Optimization (default) or Random Search. If set to Bayesian Optimization, samples new policies by a Random Forest Estimator and Expected Improvement acquisition function.

<img width="600" alt="simplified_workflow" src="https://user-images.githubusercontent.com/14996155/52587711-797a4280-2def-11e9-84f8-2368fd709ab9.png">

### Why Bayesian Optimization?

In hyperparameter optimization, main choices are random search, grid search, bayesian optimization (BO), and reinforcement learning (RL) (in the order of method complexity). Google's [AutoAugment](https://arxiv.org/abs/1805.09501) uses RL for data augmentation hyperparameter tuning, but it takes 15,000 iterations to learn policies (which means training the child CNN model 15,000 times). Thus, it requires massive computational resources. Bayesian Optimization on the other hand learns good polices in 100-300 iterations, making it +40X faster. Additionally, it is better than grid search and random search in terms of accuracy, cost, and computation time in hyperparameter tuning([ref](https://mlconf.com/lets-talk-bayesian-optimization/)) (we can think optimization of augmentation policies as a hyperparameter tuning problem where hyperparameters are concerning with augmentations instead of the deep learning architecture). This result is not surprising since despite Grid Search or Random Search BO selects new hyperparameter as informed with previous results for tried hyperparameters.

<img width="500" alt="optimization-comparison" src="https://user-images.githubusercontent.com/14996155/53222123-4ae73d80-3621-11e9-9457-44e76012d11c.png">

### How does Bayesian Optimization work?

Aim of Bayesian Optimization (BO) is finding **set of parameters** which maximize the value of an **objective function**. It builds a surrogate model for predicting value of objective function for unexplored parameters. Working cycle of BO can be summarized as:
1. Build a surrogate model of the objective function
2. Find parameters that perform best on the surrogate (or pick random hyperparameters)
3. Execute objective function with these parameters
4. Update the surrogate model with these parameters and result (value) of objective function
5. Repeat steps 2-4 until maximum number of iterations reached

For more detailed explanation, read [this blogpost](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f) explaining BO in high-level, or take a glance at [this review paper](https://ieeexplore.ieee.org/document/7352306)

### Augmentation policy

A policy describes the augmentation will be applied on a dataset. Each policy consists variables for two augmentation types, their magnitude and the portion of the data to be augmented. An example policy is as following:

<img width="400" alt="example policy" src="https://user-images.githubusercontent.com/14996155/52595719-59ed1500-2e03-11e9-9a40-a79462006924.png">

We use 26 types of transforms (from torchvison v2). They are organized by category as below:

**Geometric** (8): `rotate`, `flip_h`, `flip_v`, `affine`, `shear`, `perspective`, `elastic`, `random_crop`

**Color** (5): `brightness`, `contrast`, `saturation`, `hue`, `color_jitter`

**Advanced Color** (7): `sharpen`, `autocontrast`, `equalize`, `invert`, `solarize`, `posterize`, `grayscale`

**Blur & Noise** (2): `blur`, `gaussian_noise`

**Occlusion** (2): `erasing`, `cutout`

**Advanced** (2): `channel_permute`, `photometric_distort`

### Child model
[source](https://github.com/barisozmen/deepaugment/blob/master/deepaugment/childcnn.py#L232-L269)

Child model is trained over and over from scratch during the optimization process. Its number of training depends on the number of iterations chosen by the user, which is expected to be around 100-300 for obtaining good results. Child model is therefore the computational bottleneck of the algorithm. With the current design, training time is ~30 seconds for 32x32 images on AWS instance p3.x2large using V100 GPU (112 TensorFLOPS). It has 1,250,858 trainable parameters for 32x32 images. Below is the diagram of child model:
<img width="800" alt="child-cnn" src="https://user-images.githubusercontent.com/14996155/52545277-10e98200-2d6b-11e9-9639-48b671711eba.png">

#### Other choices for child CNN model
Standard Child model is a basic CNN where its diagram and details given above. However, you are not limited with that model. You can use your own keras model by assigning it into config dictionary as:
```Python
my_config = {"model": my_keras_model_object}
deepaug = DeepAugment(my_images, my_labels, my_config)
```
Or use an implemented small model, such as WideResNet-40-2 (while it is bigger than Basic CNN):
```Python
my_config = {"model": "wrn_40_2"} # depth(40) and wideness-factor(2) can be changed. e.g. wrn_20_4
```
Or use a big model (not recommended unless you have massive computational resources):
```Python
my_config = {"model": "InceptionV3"}
```
```Python
my_config = {"model": "MobileNetV2"}
```

### Reward function
[source](https://github.com/barisozmen/deepaugment/blob/master/deepaugment/objective.py#L69-L89)

Reward function is calculated as mean of K highest validation accuracies of the child model which is not smaller than corresponding training accuracy by 0.05. K can be determined by the user by updating `opt_last_n_epochs` key in config as argument to `DeepAugment()` class (K is 3 by default).

## Configuration

### DeepAugment Initialization

```python
DeepAugment(
    # Data
    X_train, y_train, 
    X_val, y_val,
    
    # Essential
    model="simple",              # Model architecture
    device="auto",               # "auto", "cuda", "mps", "cpu"
    random_state=42,             # Reproducibility seed
    
    # Useful
    method="bayesian",           # "bayesian" or "random"
    save_history=True,           # Save optimization history
    
    # Advanced
    transform_categories=None,   # Filter transforms by category
    custom_reward_fn=None,       # Custom reward function
    
    # Core
    n_operations=4,              # Transforms per policy
    train_size=2000,             # Training subset size
    val_size=500,                # Validation subset size
)
```

### Optimization Parameters

```python
aug.optimize(
    iterations=50,         # Policies to try
    epochs=10,            # Training epochs per policy
    samples=1,            # Runs per policy (for averaging)
    batch_size=64,        # Training batch size
    learning_rate=0.001,  # Learning rate
    early_stopping=False, # Enable early stopping
    patience=10,          # Early stopping patience
    verbose=True,         # Show progress
)
```

### Available Models

- **`"simple"`** - SimpleCNN (default, fast, 1.2M parameters)

### Transform Categories

You can restrict augmentations by category via `transform_categories`. If it is not given, then all transformations will be used.

```python
# Use only geometric transforms
aug = DeepAugment(..., transform_categories=["geometric"])

# Multiple categories
aug = DeepAugment(..., transform_categories=["geometric", "color"])
```

Categories: `geometric`, `color`, `advanced_color`, `blur_noise`, `occlusion`, `advanced`

See [augment.py](src/deepaugment/augment.py) for all available transforms.

## Data pipeline
<img width="600" alt="data-pipeline-2" src="https://user-images.githubusercontent.com/14996155/52740938-0d334680-2f89-11e9-8d68-117d139d9ab8.png">
<img width="600" alt="data-pipeline-1" src="https://user-images.githubusercontent.com/14996155/52740937-0c9ab000-2f89-11e9-9e94-beca71caed41.png">


## Development

**Contributing?** See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and workflow.

### Version Management

**Single Source of Truth**: Version lives ONLY in [pyproject.toml](pyproject.toml:3).

We use [semantic versioning](https://semver.org/) (MAJOR.MINOR.PATCH).


### Setup for Developers

First time setup:
```bash
make setup  # Installs native git pre-commit hook for auto-versioning
```

This creates a git pre-commit hook that automatically bumps patch version on every commit.

## Code Visualization
Created by [pyreverse](https://pylint.readthedocs.io/en/latest/additional_tools/pyreverse/index.html)

### Classes Diagram
<img width="1461" height="1175" alt="classes_Deepaugment" src="https://github.com/user-attachments/assets/990c809b-eb55-40d5-a896-6312f656a56c" />

### Packages Diagram
<img width="1457" height="539" alt="packages_Deepaugment-1" src="https://github.com/user-attachments/assets/8b6d0f5a-d363-4fc9-88c9-6d9e5ae3012e" />


## References
[1] Cubuk et al., 2018. AutoAugment: Learning Augmentation Policies from Data
([arxiv](https://arxiv.org/abs/1805.09501))

[2] Zoph et al., 2016. Neural Architecture Search with Reinforcement Learning
([arxiv](https://arxiv.org/abs/1611.01578))

[3] Shahriari et al., 2016. A review of Bayesian Optimization
([ieee](https://ieeexplore.ieee.org/document/7352306))

[4] Dewancker et al. Bayesian Optimization Primer ([white-paper](https://app.sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf))

[5] DeVries, Taylor 2017. Improved Regularization of CNN's with Cutout
([arxiv](https://arxiv.org/abs/1708.04552))

Blogs:
- A conceptual explanation of Bayesian Optimization ([towardsdatascience](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f))
- Comparison experiment: Bayesian Opt. vs Grid Search vs Random Search ([mlconf](https://mlconf.com/lets-talk-bayesian-optimization/))


Main dependencies:
- [scikit-optimize](scikit-optimize.github.io/) used for Bayesian optimization
- [torch](https://pytorch.org/) used to create neural networks
- [torchvision](https://pytorch.org/vision/stable/index.html) for image transformations


## Citation

Original DeepAugment paper:

```bibtex
@software{ozmen2019deepaugment,
  author = {Özmen, Barış},
  title = {DeepAugment: Automated Data Augmentation},
  year = {2019},
  url = {https://github.com/barisozmen/deepaugment}
}
```
