# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import pandas as pd
import numpy as np

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