# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import sys
import numpy as np
import keras


class DataOp:
    def load(dataset_name, training_set_size, validation_set_size):
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
        # reduce training dataset
        ix = np.random.choice(len(X_train), training_set_size, False)
        X_train = X_train[ix]
        y_train = y_train[ix]
        # reduce validation dataset
        ix = np.random.choice(len(X_val), validation_set_size, False)
        X_val = X_val[ix]
        y_val = y_val[ix]

        data = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}
        input_shape = X_train.shape[1:]

        return data, input_shape

    def preprocess(data):
        # normalize images from 0 to 1
        data["X_train"] = data["X_train"].astype("float32") / 255
        data["X_val"] = data["X_val"].astype("float32") / 255
        # convert labels to categorical
        data["y_train"] = keras.utils.to_categorical(data["y_train"])
        data["y_val"] = keras.utils.to_categorical(data["y_val"])
        return data
