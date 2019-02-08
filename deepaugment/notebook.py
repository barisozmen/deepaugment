# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import pandas as pd
import numpy as np


def get_folder_path(path):
    last = path.split("/")[-1]
    return path.replace(last , "")

class Notebook():

    def __init__(self, store_path):
        self.df = pd.DataFrame()
        self.store_path = store_path

    def record(self, trial_no, trial_hyperparams, sample_no, reward, history):
        new_df = pd.DataFrame(history)
        new_df["trial_no"] = trial_no
        new_df["aug1_type"] = trial_hyperparams[0]
        new_df["aug1_magnitude"] = trial_hyperparams[1]
        new_df["aug2_type"] = trial_hyperparams[2]
        new_df["aug2_magnitude"] = trial_hyperparams[3]
        new_df["portion"] = trial_hyperparams[4]
        new_df["sample_no"] = sample_no
        new_df["mean_late_val_acc"] = reward
        new_df = new_df.round(3) # round all float values to 3 decimals after point
        new_df["epoch"] = np.arange(1, len(new_df)+1)
        self.df = pd.concat([self.df, new_df])

    def save(self):
        self.df.to_csv(self.store_path, index=False)

    def add_records_from(self, notebook_path):
        notebook_df = pd.read_csv(notebook_path, comment="#")
        self.df = pd.concat([self.df, notebook_df])

    def get_top_policies(self, k):
        trial_avg_val_acc_df = (self.df.drop_duplicates(["trial_no", "sample_no"])
            .groupby("trial_no")
            .mean()["mean_late_val_acc"]
            .reset_index()
        )[["trial_no","mean_late_val_acc"]]

        x_df = pd.merge(
            self.df.drop(columns=["mean_late_val_acc"]),
            trial_avg_val_acc_df,
            on="trial_no", how="left"
        )

        x_df = x_df.sort_values("mean_late_val_acc", ascending=False)

        baseline_val_acc = x_df[x_df["portion"]==0.0]["mean_late_val_acc"].values[0]

        x_df["expected_accuracy_increase"] = x_df["mean_late_val_acc"] - baseline_val_acc

        top_df = (x_df.drop_duplicates(["trial_no"])
            .sort_values("mean_late_val_acc", ascending=False)
            [:k]
        )

        SELECT = ['trial_no', 'aug1_type', 'aug1_magnitude', 'aug2_type',
                  'aug2_magnitude', 'portion', 'mean_late_val_acc',
                  "expected_accuracy_increase"]

        top_df = top_df[SELECT]

        print(f"top-{k} policies:")
        print(top_df)

        top_df.to_csv( get_folder_path( get_folder_path(self.store_path) + "top{k}_policies.csv", index=False))
        return top_df



