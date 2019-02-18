# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import tensorflow as tf
import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # tell tensorflow not to use all resources
session = tf.Session(config=config)
keras.backend.set_session(session)

import os
import sys
from os.path import dirname, realpath
import pathlib
import logging
import click

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
parent_dir_of_file = dirname(dir_of_file)
sys.path.insert(0, dir_of_file)

# Set experiment name
import datetime

now = datetime.datetime.now()
EXPERIMENT_NAME = f"{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}"
EXPERIMENT_FOLDER_PATH = os.path.join(
    parent_dir_of_file, f"reports/experiments/{EXPERIMENT_NAME}"
)
log_path = pathlib.Path(EXPERIMENT_FOLDER_PATH)
log_path.mkdir(parents=True, exist_ok=True)


# import modules from DeepAugmenter
from controller import Controller
from objective import Objective
from childcnn import ChildCNN
from notebook import Notebook
from build_features import DataOp
from image_generator import deepaugment_image_generator
from lib.decorators import Reporter

logger = Reporter.logger


# warn user if TensorFlow does not see the GPU
from tensorflow.python.client import device_lib

if "GPU" not in str(device_lib.list_local_devices()):
    print("GPU not available!")
    logging.warning("GPU not available!")
# Note: GPU not among local devices means GPU not used for sure,
#       HOWEVER GPU among local devices does not guarantee it is used


DEFAULT_CONFIG = {
    "model": "basiccnn",
    "method": "bayesian_optimization",
    "train_set_size": 2000,
    "opt_samples": 3,
    "opt_last_n_epochs": 3,
    "opt_initial_points": 10,
    "child_epochs": 50,
    "child_first_train_epochs": 0,
    "child_batch_size": 64,
    "pre_aug_weights_path": "pre_aug_weights.h5",
    "logging": logging,
    "notebook_path": f"{EXPERIMENT_FOLDER_PATH}/notebook.csv",
}


class DeepAugment:
    """Initiliazes commponents of DeepAugment (e.g. Controller, Child-model, Notebook) and optimizes image augmentation hyperparameters

    """

    @logger(logfile_dir=EXPERIMENT_FOLDER_PATH)
    def __init__(self, images="cifar10", labels=None, config=None):
        """Initializes DeepAugment object

        Does following steps:
            1. load and preprocess data
            2. create child model
            3. create controller
            4. create notebook (for recording trainings)
            5. do initial training
            6. create objective function
            7. evaluate objective function without augmentation

        Args:
            images (numpy.array/str): array with shape (n_images, dim1, dim2 , channel_size), or a string with name of keras-dataset (cifar10, fashion_mnsit)
            labels (numpy.array): labels of images, array with shape (n_images) where each element is an integer from 0 to number of classes
            config (dict): dictionary of configurations, for updating the default config which is:
                {
                    "model": "basiccnn",
                    "method": "bayesian_optimization",
                    "train_set_size": 2000,
                    "opt_samples": 3,
                    "opt_last_n_epochs": 3,
                    "opt_initial_points": 10,
                    "child_epochs": 50,
                    "child_first_train_epochs": 0,
                    "child_batch_size": 64,
                    "pre_aug_weights_path": "pre_aug_weights.h5",
                    "logging": logging,
                    "notebook_path": f"{EXPERIMENT_FOLDER_PATH}/notebook.csv"
                }
        """
        self.config = DEFAULT_CONFIG
        if config!=None: self.config.update(config)
        self.iterated = 0  # keep tracks how many times optimizer iterated

        self._load_and_preprocess_data(images, labels)

        # define main objects
        self.child_model = ChildCNN(self.input_shape, self.num_classes, self.config)
        self.controller = Controller(self.config)
        self.notebook = Notebook(self.config)
        if self.config["child_first_train_epochs"] > 0:
            self._do_initial_training()
        self.child_model.save_pre_aug_weights()
        self.objective_func = Objective(
            self.data, self.child_model, self.notebook, self.config
        )

        self._evaluate_objective_func_without_augmentation()

    def optimize(self, iterations=300):
        """Optimize objective function hyperparameters using controller and child model

        Args:
            iterations (int): number of optimization iterations, which the child model will be run

        Returns:
            pandas.DataFrame: top policies (with highest expected accuracy increase)
        """
        # iteratate optimizer
        for trial_no in range(self.iterated + 1, self.iterated + iterations + 1):
            trial_hyperparams = self.controller.ask()
            print("trial:", trial_no, "\n", trial_hyperparams)
            f_val = self.objective_func.evaluate(trial_no, trial_hyperparams)
            self.controller.tell(trial_hyperparams, f_val)

        self.iterated += iterations  # update number of previous iterations

        self.top_policies = self.notebook.get_top_policies(20)
        self.notebook.output_top_policies()
        print("\ntop policies are:\n", self.top_policies)

        return self.top_policies

    def image_generator_with_top_policies(self, images, labels, batch_size=None):
        """

        Args:
            images (numpy.array): array with shape (N,dim,dim,channek-size)
            labels (numpy.array): array with shape (N), where each eleemnt is an integer from 0 to num_classes-1
            batch_size (int): batch size of the generator on demand
        Returns:
            generator: generator for augmented images
        """
        if batch_size is None:
            batch_size = self.config["child_batch_size"]

        top_policies_list = self.top_policies[
            ["aug1_type","aug1_magnitude",
             "aug2_type","aug2_magnitude",
             "portion"]
        ].to_dict(orient="records")

        return deepaugment_image_generator(images, labels, top_policies_list, batch_size=batch_size)


    def _load_and_preprocess_data(self, images, labels):
        """Loads and preprocesses data

        Records `input_shape`, `data`, and `num_classes` into `self

        Args:
            images (numpy.array/str): array with shape (n_images, dim1, dim2 , channel_size), or a string with name of keras-dataset (cifar10, fashion_mnsit)
            labels (numpy.array): labels of images, array with shape (n_images) where each element is an integer from 0 to number of classes
        """
        if isinstance(images, str):
            X, y, self.input_shape = DataOp.load(images)
        else:
            X, y = images, labels
        self.input_shape = X.shape[1:]
        self.data = DataOp.preprocess(X, y, self.config["train_set_size"])
        self.num_classes = DataOp.find_num_classes(self.data)

    def _do_initial_training(self):
        """Do the first training without augmentations

        Training weights will be used as based to further child model trainings
        """
        history = self.child_model.fit(
            self.data, epochs=self.config["child_first_train_epochs"]
        )
        self.notebook.record(
            -1, ["first", 0.0, "first", 0.0, "first", 0.0, 0.0], 1, None, history
        )

    def _evaluate_objective_func_without_augmentation(self):
        """Find out what would be the accuracy if augmentation are not applied
        """
        no_aug_hyperparams = ["crop", 0.0, "crop", 0.0, 0.0]
        f_val = self.objective_func.evaluate(0, no_aug_hyperparams)
        self.controller.tell(no_aug_hyperparams, f_val)


@click.command()
@click.option("--images", default="cifar10")
@click.option("--labels")
@click.option("--model", type=click.STRING, default="basiccnn")
@click.option("--method", type=click.STRING, default="bayesian_optimization")
@click.option("--train-set-size", type=click.INT, default=4000)
@click.option("--opt-iterations", type=click.INT, default=1000)
@click.option("--opt-samples", type=click.INT, default=5)
@click.option("--opt-last-n-epochs", type=click.INT, default=5)
@click.option("--opt-initial-points", type=click.INT, default=20)
@click.option("--child-epochs", type=click.INT, default=15)
@click.option("--child-first-train-epochs", type=click.INT, default=0)
@click.option("--child-batch-size", type=click.INT, default=32)
def main(
    images,
    labels,
    model,
    method,
    train_set_size,
    opt_iterations,
    opt_samples,
    opt_last_n_epochs,
    opt_initial_points,
    child_epochs,
    child_first_train_epochs,
    child_batch_size
):

    _config = {
        "model" : model,
        "method" : method,
        "train_set_size" : train_set_size,
        "opt_samples" : opt_samples,
        "opt_last_n_epochs" : opt_last_n_epochs,
        "opt_initial_points" : opt_initial_points,
        "child_epochs" : child_epochs,
        "child_first_train_epochs" : child_first_train_epochs,
        "child_batch_size" : child_batch_size
    }

    deepaug = DeepAugment(images, labels, config=_config)

    best_policies = deepaug.optimize(opt_iterations)

    print(best_policies)


if __name__ == "__main__":
    main()
