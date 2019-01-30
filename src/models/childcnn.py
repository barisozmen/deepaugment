# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

from keras import models, layers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


class ChildCNN:
    def __init__(self, input_shape, batch_size, epochs, num_classes):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
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
