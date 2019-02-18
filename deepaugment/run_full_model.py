# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import pathlib
import logging
import os
import datetime

import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
parent_dir_of_file = dirname(dir_of_file)
sys.path.insert(0, dir_of_file)

now = datetime.datetime.now()
EXPERIMENT_NAME = (
    f"wrn_28_10_{now.year}-{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}"
)

EXPERIMENT_FOLDER_PATH = os.path.join(
    parent_dir_of_file, f"reports/experiments/{EXPERIMENT_NAME}"
)
log_path = pathlib.Path(EXPERIMENT_FOLDER_PATH)
log_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=(log_path / "info.log").absolute(), level=logging.DEBUG)

from childcnn import ChildCNN
from build_features import DataOp
from lib.decorators import Reporter
from image_generator import deepaugment_image_generator

logger = Reporter.logger

from keras.callbacks import CSVLogger


import click

from sklearn.model_selection import train_test_split

import logging


# @click.command()
# @click.option("--dataset-name", type=click.STRING, default="cifar10")
# @click.option("--num-classes", type=click.INT, default=10)
# @click.option("--epochs", type=click.INT, default=15)
# @click.option("--batch-size", type=click.INT, default=64)
# @click.option("--policies-path", type=click.STRING, default="dont_augment")
@logger(logfile_dir=EXPERIMENT_FOLDER_PATH)
def run_full_model(images, labels, epochs, batch_size, policies_path):


    data={}
    data["X_train"], data["X_val"], data["y_train"], data["y_val"] = train_test_split(images, labels, test_size=0.30, shuffle=True)

    data = DataOp.preprocess_normal(data)

    input_shape = data["X_train"][0].shape
    num_classes = data["y_train"].shape[1]

    cnn_config={
        "model" : "inceptionv3",
        "weights" : "imagenet",
        "input_shape" : input_shape,
        "batch_size" : batch_size,
        "pre_augmentation_weights_path" : "initial_model_weights.h5",
        "logging" : logging
    }

    full_model = ChildCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        config=cnn_config
    )

    if policies_path == "dont_augment":
        policy_str = "non_augmented"
    else:
        policy_str = "augmented"
    csv_logger = CSVLogger(
        f"{EXPERIMENT_FOLDER_PATH}/wrn_28_10_training_on_{policy_str}.csv"
    )

    if policies_path == "dont_augment":
        history = full_model.fit_normal(data, epochs=epochs, csv_logger=csv_logger)
        print(f"Reached validation accuracy is {history['val_acc'][-1]}")
    else:

        datagen = deepaugment_image_generator(
            data["X_train"],
            data["y_train"],
            policies_path,
            batch_size=batch_size,
            augment_chance=0.8,
        )
        print("fitting the model")
        history = full_model.fit_with_generator(
            datagen,
            data["X_val"],
            data["y_val"],
            train_data_size=len(data["X_train"]),
            epochs=epochs,
            csv_logger=csv_logger,
        )
        print(f"Reached validation accuracy is {history['val_acc'][-1]}")


if __name__ == "__main__":

    run_full_model()
