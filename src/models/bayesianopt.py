# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

# Set experiment name
import datetime

now = datetime.datetime.now()
EXPERIMENT_NAME = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}"

import sys
import pandas as pd
import numpy as np
from skopt import gp_minimize

# Import machine learning libraries
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # tell tensorflow not to use all resources
session = tf.Session(config=config)
import keras

keras.backend.set_session(session)

import pathlib
import logging

log_path = pathlib.Path(f"../../reports/experiments/{EXPERIMENT_NAME}")
log_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=(log_path / "info.log").absolute(), level=logging.DEBUG)

# import modules from DeepAugmenter
from augmenter import Augmenter
from childcnn import ChildCNN

sys.path.insert(0, "..")
from features.build_features import DataOp

import click


@click.command()
@click.option("--dataset-name", type=click.STRING, default="cifar10")
@click.option("--num-classes", type=click.INT, default=10)
@click.option("--training-set-size", type=click.INT, default=4000)
@click.option("--validation-set-size", type=click.INT, default=1000)
@click.option("--opt-iterations", type=click.INT, default=1000)
@click.option("--opt-random-states", type=click.INT, default=20)
@click.option("--child-epochs", type=click.INT, default=120)
@click.option("--child-batch-size", type=click.INT, default=128)
def run_bayesianopt(
    dataset_name,
    num_classes,
    training_set_size,
    validation_set_size,
    opt_iterations,
    opt_random_states,
    child_epochs,
    child_batch_size,
):
    data, input_shape = DataOp.load(
        dataset_name, training_set_size, validation_set_size
    )
    data = DataOp.preprocess(data)

    augmenter = Augmenter()

    def objective_function(params):
        """Objective function for bayesian optimization

        It runs following steps:
            1. creates augmented-data by given parameters (params)
            2. creates the child model, which is a very small CNN
            3. trains the child model from scratch
            4. returns 1 – max_val_accuracy

        Args:
            params (list): first element [0] is an integer value from 1 to 4, where each represents one of transformation
                           types: Crop, GaussianBlue, Rotate, Shear. Second element [1] is magnitude of the
                           transformation
        Returns:
            float: value of objective function for given parameters
        """
        logging.info(f"In objective function with params: {params}")
        print(f"In objective function with params: {params}")

        augmented_data = augmenter.run(data["X_train"], data["y_train"], params)

        notebook_df = pd.DataFrame()

        last_5_val_acc = []
        for k in ["a", "b", "c", "d", "e"]:
            child_model = ChildCNN(
                input_shape, child_batch_size, child_epochs, num_classes
            )

            record = child_model.model.fit(
                x=np.concatenate((data["X_train"], augmented_data["X_train"]), axis=0),
                y=np.concatenate((data["y_train"], augmented_data["y_train"]), axis=0),
                batch_size=child_batch_size,
                epochs=child_epochs,
                validation_data=(data["X_val"], data["y_val"]),
                shuffle=True,
                verbose=2,
            )
            # reason I am putting history into dataframe is I want to keep record of it in the future
            history_df = pd.DataFrame(record.history).round(3)
            last_5_val_acc.append(history_df["val_acc"].tail(5).mean())
            history_df["params_0"] = params[0]
            history_df["params_1"] = params[1]
            history_df["sample"] = k
            notebook_df = pd.concat([notebook_df, history_df])
            del child_model

        # save notebook at each iteration, in case the optimization interrupted
        notebook_df.to_csv(
            f"../../reports/experiments/{EXPERIMENT_NAME}/notebook_params_{params[0]}_{params[1]}.csv",
            index=False,
        )
        return_val = 1 - np.mean(last_5_val_acc)
        logging.info(f"Objective function value is {return_val}")
        print(f"Objective function value is {return_val}")
        return return_val

    # search space to optimize for, it has currently two dimensions:
    #    1. selection of transformation type [1,4]
    #    2. magnitude of the transformation
    search_space = [
        (1, 4),  # Crop, GaussianBlur, rotate, shear
        (0.0, 1.0),  # magnitude
    ]

    # run Bayesian optimization using Gaussian Process
    result = gp_minimize(
        objective_function,
        search_space,
        n_calls=opt_iterations,
        random_state=opt_random_states,
    )
    print(result)
    logging.info(result)


if __name__ == "__main__":
    run_bayesianopt()
