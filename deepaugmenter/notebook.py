# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import pandas as pd

class Notebook():

    def __init__(self, store_path):
        self.df = pd.DataFrame()
        self.store_path = store_path

    def record(self, trial_no, trial_hyperparams, sample_no, cost, history):
        new_df = pd.DataFrame(history)
        new_df["trial_no"] = trial_no
        new_df["aug_type"] = trial_hyperparams[0]
        new_df["magnitude"] = trial_hyperparams[1]
        new_df["sample_no"] = sample_no
        new_df["mean_late_val_acc"] = cost
        self.df = pd.concat([self.df, new_df])

    def save(self):
        self.df.to_csv(self.store_path, index=False)

    def add_records_from(self, notebook_path):
        notebook_df = pd.read_csv(notebook_path, comment="#")
        self.df = pd.concat([self.df, notebook_df])