# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import pandas as pd
import numpy as np


def get_folder_path(path):
    last = path.split("/")[-1]
    return path.replace(last, "")


class Notebook:
    def __init__(self, config):
        self.df = pd.DataFrame()
        self.store_path = config["notebook_path"]

    def record(self, trial_no, trial_hyperparams, sample_no, reward, history):
        """Records one complete training of child model

        Args:
            trial_no (int): no of trial (iteration) of training
            trial_hyperparams (list) : list of data augmentation hyperparameters used for training
            sample_no (int): sample no among training with same hyperparameters
            reward (float): reward is basically last n validation accuracy before overfitting
            history (dict): history returned by keras.model.fit()
        """
        new_df = pd.DataFrame(history)
        new_df["trial_no"] = trial_no

        new_df["A_aug1_type"] = trial_hyperparams[0]
        new_df["A_aug1_magnitude"] = trial_hyperparams[1]
        new_df["A_aug2_type"] = trial_hyperparams[2]
        new_df["A_aug2_magnitude"] = trial_hyperparams[3]

        new_df["B_aug1_type"] = trial_hyperparams[4]
        new_df["B_aug1_magnitude"] = trial_hyperparams[5]
        new_df["B_aug2_type"] = trial_hyperparams[6]
        new_df["B_aug2_magnitude"] = trial_hyperparams[7]

        new_df["C_aug1_type"] = trial_hyperparams[8]
        new_df["C_aug1_magnitude"] = trial_hyperparams[9]
        new_df["C_aug2_type"] = trial_hyperparams[10]
        new_df["C_aug2_magnitude"] = trial_hyperparams[11]

        new_df["D_aug1_type"] = trial_hyperparams[12]
        new_df["D_aug1_magnitude"] = trial_hyperparams[13]
        new_df["D_aug2_type"] = trial_hyperparams[14]
        new_df["D_aug2_magnitude"] = trial_hyperparams[15]

        new_df["E_aug1_type"] = trial_hyperparams[16]
        new_df["E_aug1_magnitude"] = trial_hyperparams[17]
        new_df["E_aug2_type"] = trial_hyperparams[18]
        new_df["E_aug2_magnitude"] = trial_hyperparams[19]

        new_df["sample_no"] = sample_no
        new_df["mean_late_val_acc"] = reward
        new_df = new_df.round(3)  # round all float values to 3 decimals after point
        new_df["epoch"] = np.arange(1, len(new_df) + 1)
        self.df = pd.concat([self.df, new_df])

    def save(self):
        self.df.to_csv(self.store_path, index=False)

    def add_records_from(self, notebook_path):
        notebook_df = pd.read_csv(notebook_path, comment="#")
        self.df = pd.concat([self.df, notebook_df])

    def get_top_policies(self, k):
        """Prints and returns top-k policies

        Policies are ordered by their expected accuracy increas
        Args:
            k (int) top-k
        Returns
            pandas.DataFrame: top-k policies as dataframe
        """
        trial_avg_val_acc_df = (
            self.df.drop_duplicates(["trial_no", "sample_no"])
            .groupby("trial_no")
            .mean()["mean_late_val_acc"]
            .reset_index()
        )[["trial_no", "mean_late_val_acc"]]

        x_df = pd.merge(
            self.df.drop(columns=["mean_late_val_acc"]),
            trial_avg_val_acc_df,
            on="trial_no",
            how="left",
        )

        x_df = x_df.sort_values("mean_late_val_acc", ascending=False)

        baseline_val_acc = x_df[x_df["trial_no"] == 0]["mean_late_val_acc"].values[0]

        x_df["expected_accuracy_increase(%)"] = (
            x_df["mean_late_val_acc"] - baseline_val_acc
        )*100

        self.top_df = x_df.drop_duplicates(["trial_no"]).sort_values(
            "mean_late_val_acc", ascending=False
        )[:k]

        SELECT = [
            "trial_no",
            'A_aug1_type', 'A_aug1_magnitude', 'A_aug2_type', 'A_aug2_magnitude',
            'B_aug1_type', 'B_aug1_magnitude', 'B_aug2_type', 'B_aug2_magnitude',
            'C_aug1_type', 'C_aug1_magnitude', 'C_aug2_type', 'C_aug2_magnitude',
            'D_aug1_type', 'D_aug1_magnitude', 'D_aug2_type', 'D_aug2_magnitude',
            'E_aug1_type', 'E_aug1_magnitude', 'E_aug2_type', 'E_aug2_magnitude',
            "mean_late_val_acc", "expected_accuracy_increase(%)"
        ]
        self.top_df = self.top_df[SELECT]

        print(f"top-{k} policies:", k)
        print(self.top_df)

        return self.top_df

    def output_top_policies(self):
        k = len(self.top_df)
        out_path = get_folder_path(self.store_path) + f"top{k}_policies.csv"
        self.top_df.to_csv(out_path, index=False)
        print(f"Top policies are saved to {out_path}")
