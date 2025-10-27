# (C) 2019 Baris Ozmen <hbaristr@gmail.com>
# (C) 2024 Peter Norvig

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications

from .wide_res_net import WideResidualNetwork
from .build_features import DataOp

class ChildCNN:
    def __init__(self, input_shape=None, num_classes=None, config=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config
        self.model = self._create_child_cnn()

    def fit(self, data, augmented_data=None, epochs=None):
        epochs = epochs or self.config["child_epochs"]

        X_train = np.concatenate([data["X_train"], augmented_data["X_train"]]) if augmented_data else data["X_train"]
        y_train = np.concatenate([data["y_train"], augmented_data["y_train"]]) if augmented_data else data["y_train"]

        X_val, y_val = DataOp.sample_validation_set(data)

        return self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.config["child_batch_size"],
            epochs=epochs,
            validation_data=(X_val, y_val),
            shuffle=True,
            verbose=2,
        ).history

    def load_pre_augment_weights(self):
        self.model.load_weights(self.config["pre_aug_weights_path"])

    def save_pre_aug_weights(self):
        self.model.save_weights(self.config["pre_aug_weights_path"])

    def _create_child_cnn(self):
        model_name = self.config.get("model", "basiccnn")
        if isinstance(model_name, str):
            if model_name.lower() == "basiccnn":
                return self._build_basic_cnn()
            elif model_name.lower().startswith("wrn"):
                return self._build_wrn()
            elif model_name.lower() in ("mobilenetv2", "inceptionv3"):
                return self._build_prepared_model()
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        elif isinstance(model_name, keras.models.Model):
            return model_name
        else:
            raise TypeError(f"Unsupported model type: {type(model_name)}")

    def _build_prepared_model(self):
        model_name = self.config["model"].lower()
        weights = self.config.get("weights")

        base_model = {
            "mobilenetv2": applications.MobileNetV2,
            "inceptionv3": applications.InceptionV3,
        }[model_name](input_shape=self.input_shape, weights=weights, include_top=False)

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.7)(x)
        predictions = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        return model

    def _build_wrn(self):
        depth = int(self.config["model"].split("_")[1])
        width = int(self.config["model"].split("_")[2])
        model = WideResidualNetwork(
            depth=depth, width=width, input_shape=self.input_shape, classes=self.num_classes
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        return model

    def _build_basic_cnn(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.6),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.6),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-6),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        return model
