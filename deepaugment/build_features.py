# (C) 2019 Baris Ozmen <hbaristr@gmail.com>
# (C) 2024 Peter Norvig

import numpy as np
from tensorflow import keras

class DataOp:
    @staticmethod
    def load(dataset_name):
        try:
            (x_train, y_train), (x_val, y_val) = getattr(keras.datasets, dataset_name).load_data()
            X = np.concatenate([x_train, x_val])
            y = np.concatenate([y_train, y_val])
            return X, y, x_train.shape[1:]
        except AttributeError:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    @staticmethod
    def preprocess(X, y, train_set_size, val_set_size=1000):
        if train_set_size is None:
            train_set_size = len(X) - val_set_size

        indices = np.random.permutation(len(X))
        train_indices = indices[:train_set_size]
        val_seed_indices = indices[train_set_size:]

        data = {
            "X_train": X[train_indices].astype("float32") / 255.0,
            "y_train": keras.utils.to_categorical(y[train_indices]),
            "X_val_seed": X[val_seed_indices].astype("float32") / 255.0,
            "y_val_seed": keras.utils.to_categorical(y[val_seed_indices]),
        }
        return data

    @staticmethod
    def sample_validation_set(data, sample_size=1000):
        val_seed_size = len(data["X_val_seed"])
        indices = np.random.choice(range(val_seed_size), min(val_seed_size, sample_size), replace=False)
        return data["X_val_seed"][indices], data["y_val_seed"][indices]

    @staticmethod
    def find_num_classes(data):
        return data["y_train"].shape[1]
