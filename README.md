# DeepAugment

<img width="400" alt="concise_workflow" src="https://user-images.githubusercontent.com/14996155/52543808-6d47a400-2d61-11e9-8df7-8271872ba0ad.png">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

DeepAugment discovers best augmentation strategies tailored for your images. It optimizes augmentation hyperparameters using Bayesian Optimization, which is widely used for hyperparameter tuning. The tool:
- boosts deep learning model accuracy 5% compared to models not using augmentation.
- saves times by automating the process


Resources: [slides](https://docs.google.com/presentation/d/1toRUTT9X26ACngr6DXCKmPravyqmaGjy-eIU5cTbG1A/edit#slide=id.g4cc092dbc6_0_0)

## Installation/Usage
```console
$ pip install deepaugment
```


Simple usage (with any dataset)
```Python
from deepaugment import DeepAugment

deepaug = DeepAugment(my_data, my_labels)

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
deepaug = DeepAugment(data=x_train, labels=y_train, config=my_config)

best_policies = deepaug.optimize(300)
```


## Results
### CIFAR-10 best policies tested on WRN-28-10 
- Method: Wide-ResNet-28-10 trained with CIFAR-10 augmented images by best found policies, and with unaugmented images (everything else same).
- Result: **5.2% accuracy increase** by DeepAugment

<img src="https://user-images.githubusercontent.com/14996155/52544784-e0541900-2d67-11e9-93db-0b8b192f5b37.png" width="400"> <img src="https://user-images.githubusercontent.com/14996155/52545044-63c23a00-2d69-11e9-9879-3d7bcb8f88f4.png" width="400">
 
## How it works?

![alt text](/reports/figures/simplified_workflow.png "Workflow")

DeepAugment working method can be dissected into three areas:
1. Search space of augmentation
2. Optimizer
3. Child model

### 1. Search space of augmentation
### 2. Optimizer
### 3. Child model
<img width="800" alt="child-cnn" src="https://user-images.githubusercontent.com/14996155/52545277-10e98200-2d6b-11e9-9639-48b671711eba.png">


### Repo Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Contact
Baris Ozmen, hbaristr@gmail.com
