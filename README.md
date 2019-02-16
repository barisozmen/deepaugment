# DeepAugment

![pypi](https://img.shields.io/pypi/v/deepaugment.svg?style=flat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

DeepAugment discovers optimized augmentation strategies tailored for your images. It uses Bayesian Optimization for optimizing hyperparameters for augmentation. The tool:
1. boosts deep learning model accuracy (shown 5.2% accuracy increase (36% decrease in error) for CIFAR-10 on WRN-28-10 compared to no augmentation)
2. saves times by automating the process

Resources: [slides](https://docs.google.com/presentation/d/1toRUTT9X26ACngr6DXCKmPravyqmaGjy-eIU5cTbG1A/edit#slide=id.g4cc092dbc6_0_0)

## Installation/Usage
```console
$ pip install deepaugment
```

Simple usage (with any dataset)
```Python
from deepaugment.deepaugment import DeepAugment

deepaug = DeepAugment(my_images, my_labels)

best_policies = deepaug.optimize(300)
```

Simple usage (with cifar-10 dataset)
```Python
deepaug = DeepAugment("cifar10")

best_policies = deepaug.optimize(300)
```

Advanced usage (by changing configurations, and with fashion-mnist dataset)
```Python
from keras.datasets import fashion_mnist

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
# X_train.shape -> (N, M, M, 3)
# y_train.shape -> (N)
deepaug = DeepAugment(iamges=x_train, labels=y_train, config=my_config)

best_policies = deepaug.optimize(300)
```

## Results
### CIFAR-10 best policies tested on WRN-28-10 
- Method: Wide-ResNet-28-10 trained with CIFAR-10 augmented images by best found policies, and with unaugmented images (everything else same).
- Result: **5.2% accuracy increase** by DeepAugment

<img src="https://user-images.githubusercontent.com/14996155/52544784-e0541900-2d67-11e9-93db-0b8b192f5b37.png" width="400"> <img src="https://user-images.githubusercontent.com/14996155/52545044-63c23a00-2d69-11e9-9879-3d7bcb8f88f4.png" width="400">
 
## How it works

Package consists three main components: controller, augmenter, and child model. Overal workflow is that controller samples new augmentation policies, augmenter transforms images by the new policy, and child model is trained from scratch by augmented images. Then, a reward is calculated from child model's validation accuracy curve by the formula as explained at (reward function section). This reward is returned back to controller, and it updates its internal and samples a new augmentation policy, returning to the beginning of the cycle.

Controller might be set to use Bayesian Optimization (defaul), or Random Search. If Bayesian Optimization set, it samples new policies by a Random Forest Estimator.

<img width="600" alt="simplified_workflow" src="https://user-images.githubusercontent.com/14996155/52587711-797a4280-2def-11e9-84f8-2368fd709ab9.png">

### Augmentation policy

A policy describes the augmentation will be applied on a dataset. Each policy consists variables for two augmentation types, their magnitude and the portion of the data to be augmented. An example policy is as following: 

<img width="400" alt="example policy" src="https://user-images.githubusercontent.com/14996155/52595719-59ed1500-2e03-11e9-9a40-a79462006924.png">

There are currently 20 types of augmentation techniques (above, right) that each aug. type variable can take. All techniques are (this list might grow in later versions):
```Python
AUG_TYPES = [ "crop", "gaussian-blur", "rotate", "shear", "translate-x", "translate-y", "sharpen", "emboss", "additive-gaussian-noise", "dropout", "coarse-dropout", "gamma-contrast", "brighten", "invert", "fog", "clouds", "add-to-hue-and-saturation", "coarse-salt-pepper", "horizontal-flip", "vertical-flip"]
```
### Child model
<img width="800" alt="child-cnn" src="https://user-images.githubusercontent.com/14996155/52545277-10e98200-2d6b-11e9-9639-48b671711eba.png">

### Reward function
Reward function is calculated as mean of K highest validation accuracies of the child model which is not smaller than corresponding training accuracy by 0.05. K can be determined by the user by updating `opt_last_n_epochs` key in config dictionary as argument to `DeepAugment()` class (K is 3 by default).

## Data pipeline
<img width="600" alt="data-pipeline-2" src="https://user-images.githubusercontent.com/14996155/52740938-0d334680-2f89-11e9-8d68-117d139d9ab8.png">
<img width="600" alt="data-pipeline-1" src="https://user-images.githubusercontent.com/14996155/52740937-0c9ab000-2f89-11e9-9e94-beca71caed41.png">

## Class diagram
![classes_deepaugment](https://user-images.githubusercontent.com/14996155/52743629-4969a580-2f8f-11e9-8eb2-35aa1af161bb.png)

## Package diagram
<img width="600" alt="package-diagram" src="https://user-images.githubusercontent.com/14996155/52743630-4a023c00-2f8f-11e9-9b12-32b2ded6071b.png">
--------

## Contact
Baris Ozmen, hbaristr@gmail.com
