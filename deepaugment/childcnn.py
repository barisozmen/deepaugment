# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

from keras import optimizers, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3

import numpy as np

import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
sys.path.insert(0, dir_of_file)

from wide_res_net import WideResidualNetwork
from build_features import DataOp
from lib.decorators import Reporter
from lib.helpers import log_and_print

timer = Reporter.timer


class ChildCNN:
    """Child CNN model which reflects full models

    """

    def __init__(self, input_shape=None, num_classes=None, config=None):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config
        self.model = self.create_child_cnn()

    @timer
    def fit(self, data, augmented_data=None, epochs=None):
        """Fits the model with given data and augmented-data

        Args:
             data (dict): should have keys 'X_train' and 'y_train'
             augmented_data (dict): should have keys 'X_train' and 'y_train'. If none, augmented-data not used
             epochs (int):
        Returns:
            dict: history of training
        """

        if epochs is None:
            epochs = self.config["child_epochs"]

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
            batch_size=self.config["child_batch_size"],
            epochs=epochs,
            validation_data=(X_val, y_val),
            shuffle=True,
            verbose=2,
        )
        return record.history

    @timer
    def fit_normal(self, data, epochs=None, csv_logger=None):
        record = self.model.fit(
            x=data["X_train"],
            y=data["y_train"],
            batch_size=self.config["child_batch_size"],
            epochs=epochs,
            validation_data=(data["X_val"], data["y_val"]),
            shuffle=True,
            verbose=2,
            callbacks=[csv_logger],
        )
        return record.history

    def fit_with_generator(
        self, datagen, X_val, y_val, train_data_size, epochs=None, csv_logger=None
    ):
        record = self.model.fit_generator(
            datagen,
            validation_data=(X_val, y_val),
            steps_per_epoch=train_data_size // self.config["child_batch_size"],
            epochs=epochs,
            shuffle=True,
            verbose=2,
            callbacks=[csv_logger],
        )
        return record.history

    @timer
    def load_pre_augment_weights(self):
        self.model.load_weights(self.config["pre_aug_weights_path"])

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
        log_and_printt(f"Test loss:{test_loss}")
        log_and_print(f"Test accuracy:{test_acc}")
        return test_loss, test_acc

    def save_pre_aug_weights(self):
        self.model.save_weights(self.config["pre_aug_weights_path"])

    def create_child_cnn(self):
        """Creates the child CNN

        Model choices:
            basicCNN
            WRN (with any N and k)
            MobileNet
        """
        if isinstance(self.config["model"], str):
            if self.config["model"].lower() == "basiccnn":
                return self.build_basicCNN()
            elif self.config["model"].lower().startswith("wrn"):
                return self.build_wrn()
            elif self.config["model"].lower() in ("mobilenetv2","inceptionv3"):
                return self.build_prepared_model()
            else:
                print(f"config['model'] should be any of 'basiccnn', 'wrn_?_?', 'mobilenetv2', 'inceptionv3'")
                raise ValueError
        else:  # if a keras model is the models itself
            return self.config["model"]

    def build_prepared_model(self):

        if self.config["model"].lower()=="mobilenetv2":
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                weights=self.config['weights'],
                include_top=False
            )
        elif self.config["model"].lower()=="inceptionv3":
            base_model = InceptionV3(
                input_shape=self.input_shape,
                weights=self.config['weights'],
                include_top=False
            )

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        # and a logistic layer
        predictions = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in model.layers:
            layer.trainable = True

        adam_opt = optimizers.Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False,
            clipnorm=1.0,
        )
        model.compile(
            loss="categorical_crossentropy", optimizer=adam_opt, metrics=["accuracy"]
        )
        log_and_print(
            f"{self.config['model']} model built as child model.\n Model summary:",
            self.config['logging'],
        )
        print(model.summary())
        return model

    def build_wrn(self):
        # For WRN-16-8 put N = 2, k = 8
        # For WRN-28-10 put N = 4, k = 10
        # For WRN-40-4 put N = 6, k = 4
        _depth = int(self.config["model"].split("_")[1])  # e.g. wrn_[40]_4
        _width = int(self.config["model"].split("_")[2])  # e.g. wrn_40_[4]
        model = WideResidualNetwork(
            depth=_depth,
            width=_width,
            dropout_rate=0.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=self.input_shape,
            classes=self.num_classes,
            activation="softmax",
        )

        adam_opt = optimizers.Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False,
            clipnorm=1.0,
        )
        model.compile(
            loss="categorical_crossentropy", optimizer=adam_opt, metrics=["accuracy"]
        )
        log_and_print(
            f"{self.config['model']} model built as child model.\n Model summary:",
            self.config["logging"],
        )
        print(model.summary())
        return model

    def build_basicCNN(self):
        """Builds basic convolutional neural net (CNN) model

        Returns:
            keras.models.Model

        :return:
        """
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
