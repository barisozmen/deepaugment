# (C) 2019 Baris Ozmen <hbaristr@gmail.com>
# (C) 2024 Peter Norvig

import pandas as pd
import numpy as np
import os

class Notebook:
    def __init__(self, config):
        self.df = pd.DataFrame()
        self.store_path = config["notebook_path"]

    def record(self, trial_no, trial_hyperparams, sample_no, reward, history):
        history_df = pd.DataFrame(history)
        history_df["trial_no"] = trial_no
        history_df["sample_no"] = sample_no
        history_df["reward"] = reward
        history_df["epoch"] = np.arange(1, len(history_df) + 1)

        for i, param in enumerate(trial_hyperparams):
            history_df[f"param_{i}"] = param

        self.df = pd.concat([self.df, history_df.round(3)])

    def save(self):
        self.df.to_csv(self.store_path, index=False)

    def get_top_policies(self, k):
        avg_rewards = self.df.groupby("trial_no")["reward"].mean()
        baseline_reward = avg_rewards.get(0, 0)

        top_trials = avg_rewards.nlargest(k)

        top_policies = self.df[self.df["trial_no"].isin(top_trials.index)].copy()
        top_policies["expected_accuracy_increase(%)"] = (top_policies["reward"] - baseline_reward) * 100

        self.top_df = top_policies.drop_duplicates("trial_no").sort_values("reward", ascending=False)
        return self.top_df

    def output_top_policies(self):
        k = len(self.top_df)
        folder_path = os.path.dirname(self.store_path)
        out_path = os.path.join(folder_path, f"top{k}_policies.csv")
        self.top_df.to_csv(out_path, index=False)
        print(f"Top policies are saved to {out_path}")
