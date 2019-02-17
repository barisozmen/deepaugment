# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import sys
import numpy as np
import keras


class DataOp:
    @staticmethod
    def load(dataset_name):
        """Loads dataset from keras and returns a sample out of it

        Args:
            dataset_name (str):
            training_set_size (int):
            validation_set_size (int):
        Returns:
            dict: data, with keys X_train, Y_train, X_val, Y_val
            list: input shape
        """
        if hasattr(keras.datasets, dataset_name):
            (x_train, y_train), (x_val, y_val) = getattr(
                keras.datasets, dataset_name
            ).load_data()
        else:
            sys.exit(f"Unknown dataset {dataset_name}")

        X = np.concatenate([x_train, x_val])
        y = np.concatenate([y_train, y_val])
        input_shape = x_train.shape[1:]

        return X, y, input_shape

    @staticmethod
    def preprocess_normal(data):
        # normalize images
        data["X_train"] = data["X_train"].astype("float32") / 255
        data["X_val"] = data["X_val"].astype("float32") / 255

        # convert labels to categorical
        data["y_train"] = keras.utils.to_categorical(data["y_train"])
        data["y_val"] = keras.utils.to_categorical(data["y_val"])
        return data

    @staticmethod
    def split_train_val_sets(X, y, train_set_size, val_set_size):
        """Splits given images randomly into `train` and `val_seed` groups

        val_seed -> is validation seed dataset, from where validation sets are sampled

        Args:
            X (numpy.array):
            y (numpy.array):
            train_set_size (int):
            val_set_size (int):
        return:
            dict: dict with keys `X_train`, `y_train`, `X_val_seed`, `y_val_seed`
        """
        if train_set_size == None:
            print(f"Using all training images")
            train_set_size = len(X) - val_set_size
        else:
            print(f"Using {train_set_size} training images")

        # reduce training dataset
        ix = np.random.choice(len(X), train_set_size, False)
        X_train = X[ix]
        y_train = y[ix]

        other_ix = set(np.arange(len(X))).difference(set(ix))
        other_ix = list(other_ix)
        X_val_seed = X[other_ix]
        y_val_seed = y[other_ix]

        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val_seed": X_val_seed,
            "y_val_seed": y_val_seed,
        }
        return data

    @staticmethod
    def preprocess(X, y, train_set_size, val_set_size=1000):
        """Preprocess images by:
            1. normalize to 0-1 range (divide by 255)
            2. convert labels to categorical)

        Args:
            X (numpy.array):
            y (numpy.array):
            train_set_size (int):
            val_set_size (int):

        Returns:
            dict: preprocessed data
        """

        data = DataOp.split_train_val_sets(X, y, train_set_size, val_set_size)

        # normalize images
        data["X_train"] = data["X_train"].astype("float32") / 255
        data["X_val_seed"] = data["X_val_seed"].astype("float32") / 255

        # convert labels to categorical
        data["y_train"] = keras.utils.to_categorical(data["y_train"])
        data["y_val_seed"] = keras.utils.to_categorical(data["y_val_seed"])
        return data

    @staticmethod
    def sample_validation_set(data):
        val_seed_size = len(data["X_val_seed"])
        ix = np.random.choice(range(val_seed_size), min(val_seed_size, 1000), False)
        X_val = data["X_val_seed"][ix].copy()
        y_val = data["y_val_seed"][ix].copy()
        return X_val, y_val

    @staticmethod
    def find_num_classes(data):
        return data["y_train"].shape[1]
