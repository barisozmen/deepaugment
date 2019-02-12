# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import pandas as pd
import numpy as np


import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
sys.path.insert(0, dir_of_file)

from augmenter import augment_by_policy
from lib.helpers import log_and_print


class Objective:
    """Objective class for the controller

    """
    def __init__(self, data, child_model, notebook, config):
        self.data = data
        self.child_model = child_model
        self.opt_samples = config["opt_samples"]
        self.opt_last_n_epochs = config["opt_last_n_epochs"]
        self.notebook = notebook
        self.logging = config["logging"]

    def evaluate(self, trial_no, trial_hyperparams):
        """Evaluates objective function

        Trains the child model k times with same augmentation hyperparameters.
        k is determined by the user by `opt_samples` argument.

        Args:
            trial_no (int): no of trial. needed for recording to notebook
            trial_hyperparams (list)
        Returns:
            float: trial-cost = 1 - avg. rewards from samples
        """

        augmented_data = augment_by_policy(
            self.data["X_train"], self.data["y_train"], *trial_hyperparams
        )

        sample_rewards = []
        for sample_no in range(1, self.opt_samples + 1):
            self.child_model.load_pre_augment_weights()
            # TRAIN
            history = self.child_model.fit(self.data, augmented_data)
            #
            reward = self.calculate_reward(history)
            sample_rewards.append(reward)
            self.notebook.record(
                trial_no, trial_hyperparams, sample_no, reward, history
            )

        trial_cost = 1 - np.mean(sample_rewards)
        self.notebook.save()

        log_and_print(
            f"{str(trial_no)}, {str(trial_cost)}, {str(trial_hyperparams)}",
            self.logging,
        )

        return trial_cost

    def calculate_reward(self, history):
        """Calculates reward for the history.

        Reward is mean of largest n validation accuracies which are not overfitting.
        n is determined by the user by `opt_last_n_epochs` argument. A validation
        accuracy is considered as overfitting if the training accuracy in the same
        epoch is larger by 0.05

        Args:
            history (dict): dictionary of loss and accuracy
        Returns:
            float: reward
        """
        history_df = pd.DataFrame(history)
        history_df["acc_overfit"] = history_df["acc"] - history_df["val_acc"]
        reward = (
            history_df[history_df["acc_overfit"] <= 0.05]["val_acc"]
            .nlargest(self.opt_last_n_epochs)
            .mean()
        )
        return reward
