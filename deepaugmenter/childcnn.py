# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

from keras import models, layers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import wide_residual_networks as wrn

import numpy as np

import sys
from lib.decorators import Reporter
from lib.helpers import log_and_print
timer = Reporter.timer


class ChildCNN:
    def __init__(
        self, model_name="basicCNN",
        input_shape=None, batch_size=None,
        num_classes=None, pre_augmentation_weights_path=None,
        logging=None
    ):
        self.model_name = model_name
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.pre_augmentation_weights_path = pre_augmentation_weights_path
        self.logging = logging
        self.model = self.create_child_cnn()

    def create_child_cnn(self):
        if self.model_name == "basicCNN":
            return self.build_basicCNN()
        elif self.model_name.startswith("wrn"):
            return self.build_wrn()
        else:
            raise ValueError

    def build_wrn(self):
        # For WRN-16-8 put N = 2, k = 8
        # For WRN-28-10 put N = 4, k = 10
        # For WRN-40-4 put N = 6, k = 4
        _nb_layers = int(self.model_name.split("_")[1]) # e.g. wrn_[40]_4
        _k = int(self.model_name.split("_")[2]) # e.g. wrn_40_[4]
        _N = int((_nb_layers - 4) / 6) # this formula taken from https://github.com/titu1994/Wide-Residual-Networks#usage
        model = wrn.create_wide_residual_network(
            self.input_shape, nb_classes=self.num_classes, N=_N, k=_k,
            conv_dropout=0.0, dense_dropout=0.0
        )

        adam_opt = optimizers.Adam(
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
            decay=0.0, amsgrad=False, clipnorm=1.0
        )
        model.compile(loss="categorical_crossentropy", optimizer=adam_opt, metrics=["accuracy"])
        log_and_print(f"{self.model_name} model built as child model.\n Model summary:", self.logging)
        print(model.summary())
        return model


    def build_basicCNN(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))

        optimizer = optimizers.RMSprop(lr=0.0001, decay=1e-6)
        # optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        print("BasicCNN model built as child model.\n Model summary:")
        print(model.summary())
        return model



    @timer
    def fit(self, data, augmented_data=None, epochs=None):

        if augmented_data is None:
            X_train = data["X_train"]
            y_train = data["y_train"]
        else:
            X_train = np.concatenate([data["X_train"], augmented_data["X_train"]])
            y_train = np.concatenate([data["y_train"], augmented_data["y_train"]])

        record = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=(data["X_val"], data["y_val"]),
            shuffle=True,
            verbose=2,
        )
        return record.history

    @timer
    def load_pre_augment_weights(self):
        self.model.load_weights(self.pre_augmentation_weights_path)
