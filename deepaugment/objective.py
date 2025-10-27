# (C) 2019 Baris Ozmen <hbaristr@gmail.com>
# (C) 2024 Peter Norvig

import pandas as pd
import numpy as np

from .augmenter import augment_by_policy

class Objective:
    def __init__(self, data, child_model, notebook, config):
        self.data = data
        self.child_model = child_model
        self.notebook = notebook
        self.config = config

    def evaluate(self, trial_no, trial_hyperparams):
        augmented_data = augment_by_policy(
            self.data["X_train"], self.data["y_train"], *trial_hyperparams
        )

        sample_rewards = []
        for sample_no in range(1, self.config["opt_samples"] + 1):
            self.child_model.load_pre_augment_weights()
            history = self.child_model.fit(self.data, augmented_data)
            reward = self._calculate_reward(history)
            sample_rewards.append(reward)
            self.notebook.record(trial_no, trial_hyperparams, sample_no, reward, history)

        trial_cost = 1 - np.mean(sample_rewards)
        self.notebook.save()

        print(f"{trial_no}, {trial_cost}, {trial_hyperparams}")

        return trial_cost

    def _calculate_reward(self, history):
        history_df = pd.DataFrame(history)
        # In TF2, 'acc' is 'accuracy'
        history_df["acc_overfit"] = history_df["accuracy"] - history_df["val_accuracy"]
        reward = (
            history_df[history_df["acc_overfit"] <= 0.10]["val_accuracy"]
            .nlargest(self.config["opt_last_n_epochs"])
            .mean()
        )
        return reward
