# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

from keras import models, layers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import numpy as np

import sys
from lib.decorators import Reporter
timer = Reporter.timer


class ChildCNN:
    def __init__(self, input_shape, batch_size, num_classes, pre_augmentation_weights_path):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.pre_augmentation_weights_path = pre_augmentation_weights_path
        self.model = self.create_child_cnn()

    def create_child_cnn(self):
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
