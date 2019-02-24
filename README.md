# DeepAugment

![pypi](https://img.shields.io/pypi/v/deepaugment.svg?style=flat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

DeepAugment discovers optimized augmentation strategies tailored for your images. It uses Bayesian Optimization for optimizing hyperparameters for augmentation. The tool:
1. reduces error rate of CNN models (shown 37% decrease in error for CIFAR-10 on WRN-28-10 compared to no augmentation)
2. saves times by automating the process

Resources: [slides](https://docs.google.com/presentation/d/1toRUTT9X26ACngr6DXCKmPravyqmaGjy-eIU5cTbG1A/edit#slide=id.g4cc092dbc6_0_0)

## Installation/Usage
Example: [google-colab](bit.ly/deepaugmentusage)

```console
$ pip install deepaugment
```

### Simple usage (with any dataset)
```Python
from deepaugment.deepaugment import DeepAugment

deepaug = DeepAugment(my_images, my_labels)

best_policies = deepaug.optimize(300)
```

### Simple usage (with a dataset on keras)
```Python
deepaug = DeepAugment("cifar10")

best_policies = deepaug.optimize(300)
```

### Advanced usage
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

### Why Bayesian Optimization?

In hyperparameter optimization, main choices are random search, grid search, bayesian optimization, and reinforcement learning (in the order of method complexity). Google's [AutoAugment](https://arxiv.org/abs/1805.09501) uses Reinforcement Learning for the data augmentation hyperparameter tuning, but it takes 15,000 iterations to learn policies (which means training the child CNN model 15,000 times). Thus, it requires huge computational resources. Bayesian Optimization on the other hand learns good polices in 100-300 iterations, making it +40X faster. Additionally, Bayesian Optimization beats grid search and random search in terms of accuracy, cost, and computation time ([ref](https://mlconf.com/lets-talk-bayesian-optimization/)) in hyperparameter tuning.

<img width="500" alt="optimization-comparison" src="https://user-images.githubusercontent.com/14996155/53222123-4ae73d80-3621-11e9-9457-44e76012d11c.png">

### How does Bayesian Optimization work?

Aim of Bayesian Optimization (BO) is finding **set of parameters** which maximize the value of an **objective function**. It builds a surrogate model for predicting value of objective function for unexplored parameters. Working cycle of BO can be summarized as:
1. Build a surrogate model of the objective function 
2. Find parameters that perform best on the surrogate (or pick random hyperparameters)
3. Execute objective function with these parameters
4. Update the surrogate model with these parameters and result (value) of objective function
5. Repeat steps 2-4 until maximimum number of iterations reached

For more detailed explanation, read [this blogpost](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f) explaining BO in high-level, or take a glance at [this review paper](https://ieeexplore.ieee.org/document/7352306)

### Augmentation policy

A policy describes the augmentation will be applied on a dataset. Each policy consists variables for two augmentation types, their magnitude and the portion of the data to be augmented. An example policy is as following: 

<img width="400" alt="example policy" src="https://user-images.githubusercontent.com/14996155/52595719-59ed1500-2e03-11e9-9a40-a79462006924.png">

There are currently 20 types of augmentation techniques (above, right) that each aug. type variable can take. All techniques are (this list might grow in later versions):
```Python
AUG_TYPES = [ "crop", "gaussian-blur", "rotate", "shear", "translate-x", "translate-y", "sharpen", "emboss", "additive-gaussian-noise", "dropout", "coarse-dropout", "gamma-contrast", "brighten", "invert", "fog", "clouds", "add-to-hue-and-saturation", "coarse-salt-pepper", "horizontal-flip", "vertical-flip"]
```
### Child model
[source](https://github.com/barisozmen/deepaugment/blob/master/deepaugment/childcnn.py#L232-L269)

Child model is trained over and over from scratch during the optimization process. Its number of training depends on the number of iterations chosen by the user, which is expected to be around 100-300 for obtaining good results. Child model is therefore the computational bottleneck of the algorithm. With the current design, training time is ~30 seconds for 32x32 images on AWS instance p3.x2large using V100 GPU (112 TensorFLOPS). Below is the diagram of child model:
<img width="800" alt="child-cnn" src="https://user-images.githubusercontent.com/14996155/52545277-10e98200-2d6b-11e9-9639-48b671711eba.png">

### Reward function
[source](https://github.com/barisozmen/deepaugment/blob/master/deepaugment/objective.py#L69-L89)

Reward function is calculated as mean of K highest validation accuracies of the child model which is not smaller than corresponding training accuracy by 0.05. K can be determined by the user by updating `opt_last_n_epochs` key in config as argument to `DeepAugment()` class (K is 3 by default).

## Data pipeline
<img width="600" alt="data-pipeline-2" src="https://user-images.githubusercontent.com/14996155/52740938-0d334680-2f89-11e9-8d68-117d139d9ab8.png">
<img width="600" alt="data-pipeline-1" src="https://user-images.githubusercontent.com/14996155/52740937-0c9ab000-2f89-11e9-9e94-beca71caed41.png">

## Class diagram
![classes_deepaugment](https://user-images.githubusercontent.com/14996155/52743629-4969a580-2f8f-11e9-8eb2-35aa1af161bb.png)

## Package diagram
<img width="600" alt="package-diagram" src="https://user-images.githubusercontent.com/14996155/52743630-4a023c00-2f8f-11e9-9b12-32b2ded6071b.png">

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
    
Libraries:
- [scikit-optimize](scikit-optimize.github.io/)
- [mgaug](github.com/aleju/imgaug)
- [AutoAugment-unofficial](github.com/barisozmen/autoaugment-unofficial)
- [Automold]() (Self-driving car image-augmentation library)

--------

## Contact
Baris Ozmen, hbaristr@gmail.com
