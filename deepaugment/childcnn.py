# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

from keras import models, layers, optimizers, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.applications.mobilenetv2 import MobileNetV2

from wide_res_net import WideResidualNetwork
from build_features import DataOp

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

    @timer
    def fit_normal(self, data, epochs=None, csv_logger=None):
        record = self.model.fit(
            x=data["X_train"],
            y=data["y_train"],
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=(data["X_val"], data["y_val"]),
            shuffle=True,
            verbose=2,
            callbacks=[csv_logger]
        )
        return record.history

    def fit_with_generator(self, datagen, X_val, y_val, train_data_size, epochs=None, csv_logger=None):
        record = self.model.fit_generator(
            datagen, validation_data=(X_val, y_val),
            steps_per_epoch=train_data_size//self.batch_size,
            epochs=epochs,
            shuffle=True,
            verbose=2,
            callbacks=[csv_logger]
        )
        return record.history


    @timer
    def fit(self, data, augmented_data=None, epochs=None):

        if augmented_data is None:
            X_train = data["X_train"]
            y_train = data["y_train"]
        else:
            X_train = np.concatenate([data["X_train"], augmented_data["X_train"]])
            y_train = np.concatenate([data["y_train"], augmented_data["y_train"]])

        X_val, y_val = DataOp.sample_validation_set(data)

        record = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            shuffle=True,
            verbose=2,
        )
        return record.history

    @timer
    def load_pre_augment_weights(self):
        self.model.load_weights(self.pre_augmentation_weights_path)

    def evaluate_with_refreshed_validation_set(self, data):
        X_val_backup = data["X_val_backup"]
        y_val_backup = data["y_val_backup"]
        # FIXME if dataset is smaller than 5000, an error will occur
        ivb = np.random.choice(len(X_val_backup), 5000, False)
        X_val_backup = X_val_backup[ivb]
        y_val_backup = y_val_backup[ivb]

        scores = self.model.evaluate(X_val_backup, y_val_backup, verbose=2)

        test_loss = scores[0]
        test_acc = scores[1]
        log_and_printt(f'Test loss:{test_loss}')
        log_and_print(f'Test accuracy:{test_acc}')
        return test_loss, test_acc

    def create_child_cnn(self):
        if self.model_name.lower() == "basiccnn":
            return self.build_basicCNN()
        elif self.model_name.lower().startswith("wrn"):
            return self.build_wrn()
        elif self.model_name.lower() == "mobilenet":
            return self.build_mobilenetv2()
        else:
            raise ValueError

    def build_mobilenetv2(self):
        mobilenet_v2 = MobileNetV2(
            input_shape=self.input_shape, weights=None, include_top=False
        )

        # add a global spatial average pooling layer
        x = mobilenet_v2.output
        x = GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.1)(x)
        # and a logistic layer
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=mobilenet_v2.input, outputs=predictions)

        for layer in model.layers:
            layer.trainable = True

        adam_opt = optimizers.Adam(
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
            decay=0.0, amsgrad=False, clipnorm=1.0
        )
        model.compile(loss="categorical_crossentropy", optimizer=adam_opt, metrics=["accuracy"])
        log_and_print(f"{self.model_name} model built as child model.\n Model summary:", self.logging)
        print(model.summary())
        return model

        return model

    def build_wrn(self):
        # For WRN-16-8 put N = 2, k = 8
        # For WRN-28-10 put N = 4, k = 10
        # For WRN-40-4 put N = 6, k = 4
        _depth = int(self.model_name.split("_")[1]) # e.g. wrn_[40]_4
        _width = int(self.model_name.split("_")[2]) # e.g. wrn_40_[4]
        model = WideResidualNetwork(depth=_depth, width=_width, dropout_rate=0.0,
                                    include_top=True, weights=None,
                                    input_tensor=None, input_shape=self.input_shape,
                                    classes=self.num_classes, activation='softmax')

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
        model.add(Dropout(0.6))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))

        optimizer = optimizers.RMSprop(lr=0.001, decay=1e-6)
        # optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        print("BasicCNN model built as child model.\n Model summary:")
        print(model.summary())
        return model




