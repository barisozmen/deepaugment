# (C) 2019 Baris Ozmen <hbaristr@gmail.com>
# (C) 2024 Peter Norvig

import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import os
import pathlib
import logging
from .controller import Controller
from .objective import Objective
from .childcnn import ChildCNN
from .notebook import Notebook
from .build_features import DataOp
from .image_generator import deepaugment_image_generator

# Set experiment name
now = datetime.datetime.now()
EXPERIMENT_NAME = f"{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}"
EXPERIMENT_FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), f"reports/experiments/{EXPERIMENT_NAME}"
)
pathlib.Path(EXPERIMENT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)


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
    "notebook_path": os.path.join(EXPERIMENT_FOLDER_PATH, "notebook.csv"),
}


class DeepAugment:
    def __init__(self, images="cifar10", labels=None, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.iterated = 0

        self._load_and_preprocess_data(images, labels)

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
        for trial_no in range(self.iterated + 1, self.iterated + iterations + 1):
            trial_hyperparams = self.controller.ask()
            print(f"trial: {trial_no}\n {trial_hyperparams}")
            f_val = self.objective_func.evaluate(trial_no, trial_hyperparams)
            self.controller.tell(trial_hyperparams, f_val)

        self.iterated += iterations

        self.top_policies = self.notebook.get_top_policies(20)
        self.notebook.output_top_policies()
        print("\ntop policies are:\n", self.top_policies)

        return self.top_policies

    def image_generator_with_top_policies(self, images, labels, batch_size=None):
        batch_size = batch_size or self.config["child_batch_size"]
        policies = self.top_policies.to_dict(orient="records")
        return deepaugment_image_generator(images, labels, policies, batch_size=batch_size)

    def _load_and_preprocess_data(self, images, labels):
        if isinstance(images, str):
            X, y, self.input_shape = DataOp.load(images)
        else:
            X, y = images, labels
        self.input_shape = X.shape[1:]
        self.data = DataOp.preprocess(X, y, self.config["train_set_size"])
        self.num_classes = DataOp.find_num_classes(self.data)

    def _do_initial_training(self):
        history = self.child_model.fit(
            self.data, epochs=self.config["child_first_train_epochs"]
        )
        self.notebook.record(
            -1, ["first", 0.0, "first", 0.0, "first", 0.0, 0.0], 1, None, history
        )

    def _evaluate_objective_func_without_augmentation(self):
        no_aug_hyperparams = ["rotate", 0.0] * 10
        f_val = self.objective_func.evaluate(0, no_aug_hyperparams)
        self.controller.tell(no_aug_hyperparams, f_val)

