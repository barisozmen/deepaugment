# (C) 2024 Peter Norvig

import numpy as np
import pandas as pd
from tensorflow import keras
from .augmenter import augment_by_policy

class ImageAugmentationGenerator(keras.utils.Sequence):
    def __init__(self, X, y, policy, batch_size=64, augment_chance=0.5):
        self.X, self.y = X, y
        self.policy = self._load_policy(policy)
        self.batch_size = batch_size
        self.augment_chance = augment_chance
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch, y_batch = self.X[indices], self.y[indices]

        if np.random.rand() < self.augment_chance:
            policy_chain = self.policy[np.random.randint(len(self.policy))]
            hyperparams = list(policy_chain.values())
            augmented_data = augment_by_policy(X_batch, y_batch, *hyperparams)
            return augmented_data["X_train"], augmented_data["y_train"]
        else:
            return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)

    def _load_policy(self, policy):
        if isinstance(policy, str):
            if policy == "random":
                return [
                    {
                        "aug1_type": np.random.choice(AUG_TYPES),
                        "aug1_magnitude": np.random.rand(),
                        "aug2_type": np.random.choice(AUG_TYPES),
                        "aug2_magnitude": np.random.rand(),
                    } for _ in range(20)
                ]
            else:
                return pd.read_csv(policy).to_dict(orient="records")
        return policy
