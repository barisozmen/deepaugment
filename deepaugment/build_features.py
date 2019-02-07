# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import sys
import numpy as np
import keras


class DataOp:

    @staticmethod
    def load(dataset_name, training_set_size=None):
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
            (X_train, y_train), (X_val, y_val) = getattr(
                keras.datasets, dataset_name
            ).load_data()
        else:
            sys.exit(f"Unknown dataset {dataset_name}")
        
        input_shape = X_train.shape[1:]
        
        if training_set_size==None:
            print(f"Using all trainin images")
            data = {"X_train": X_train, "y_train": y_train,
                    "X_val": X_val, "y_val": y_val}

        else:
            print(f"Using {training_set_size} training images")
            # reduce training dataset
            ix = np.random.choice(len(X_train), training_set_size, False)
            X_train_reduced = X_train[ix]
            y_train_reduced = y_train[ix]

            other_ix = set(np.arange(len(X_train))).difference(set(ix))
            other_ix = list(other_ix)
            X_train_non_chosen = X_train[other_ix]
            y_train_non_chosen = y_train[other_ix]

            X_val_seed = np.concatenate([X_val, X_train_non_chosen])
            y_val_seed = np.concatenate([y_val, y_train_non_chosen])

            data = {"X_train": X_train_reduced, "y_train": y_train_reduced,
                    "X_val_seed": X_val_seed, "y_val_seed": y_val_seed}

        return data, input_shape

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
    def preprocess(data):
        # normalize images
        data["X_train"] = data["X_train"].astype("float32") / 255
        data["X_val_seed"] = data["X_val_seed"].astype("float32") / 255

        # convert labels to categorical
        data["y_train"] = keras.utils.to_categorical(data["y_train"])
        data["y_val_seed"] = keras.utils.to_categorical(data["y_val_seed"])
        return data

    @staticmethod
    def sample_validation_set(data):
        ix = np.random.choice(range(len(data["X_val_seed"])), 1000, False)
        X_val = data["X_val_seed"][ix].copy()
        y_val = data["y_val_seed"][ix].copy()
        return X_val, y_val
