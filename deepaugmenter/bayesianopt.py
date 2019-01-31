# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

# Set experiment name
import datetime

now = datetime.datetime.now()
EXPERIMENT_NAME = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}"

import os
import sys
import pandas as pd
import numpy as np
import skopt
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

from os.path import dirname, realpath
file_path = realpath(__file__)
dir_of_file = dirname(file_path)
parent_dir_of_file = dirname(dir_of_file)
sys.path.insert(0, parent_dir_of_file)

# import modules from DeepAugmenter
from augmenter import Augmenter
from childcnn import ChildCNN
from notebook import Notebook
notebook = Notebook(f"../../reports/experiments/{EXPERIMENT_NAME}/notebook.csv")
from features.build_features import DataOp
from lib.decorators import Reporter
logger = Reporter.logger

import click
@click.command()
@click.option("--dataset-name", type=click.STRING, default="cifar10")
@click.option("--num-classes", type=click.INT, default=10)
@click.option("--training-set-size", type=click.INT, default=4000)
@click.option("--validation-set-size", type=click.INT, default=1000)
@click.option("--opt-iterations", type=click.INT, default=1000)
@click.option("--opt-samples", type=click.INT, default=5)
@click.option("--opt-last-n-epochs", type=click.INT, default=5)
@click.option("--opt-initial-points", type=click.INT, default=20)
@click.option("--child-epochs", type=click.INT, default=15)
@click.option("--child-first-train-epochs", type=click.INT, default=0)
@click.option("--child-batch-size", type=click.INT, default=32)
@logger(logfile_dir=f"../../reports/experiments/{EXPERIMENT_NAME}")
def run_bayesianopt(
    dataset_name,
    num_classes,
    training_set_size,
    validation_set_size,
    opt_iterations,
    opt_samples,
    opt_last_n_epochs,
    opt_initial_points,
    child_epochs,
    child_first_train_epochs,
    child_batch_size,
):

    data, input_shape = DataOp.load(
        dataset_name, training_set_size, validation_set_size
    )
    data = DataOp.preprocess(data)

    child_model = ChildCNN(
        input_shape, child_batch_size, num_classes,
        "initial_model_weights.h5"
    )
    # first training
    if child_first_train_epochs>0:
        history = child_model.fit(data, epochs=child_first_train_epochs)
        notebook.record(0, ["-", "-"], 1, None, history)
    #
    child_model.model.save_weights(child_model.pre_augmentation_weights_path)
    augmenter = Augmenter()


    ####################################################################################################
    # Implementation of skopt by ask-tell design pattern
    # See https://geekyisawesome.blogspot.com/2018/07/hyperparameter-tuning-using-scikit.html
    ####################################################################################################

    opt = skopt.Optimizer(
        [
            skopt.space.Categorical(np.arange(1,9,1), name='aug_type'),
            skopt.space.Real(0.0, 1.0, name='magnitude')
        ],
        n_initial_points=opt_initial_points,
        base_estimator='RF', # Random Forest estimator
        acq_func='EI', # Expected Improvement
        acq_optimizer='auto',
        random_state=0
    )

    # skopt works with opt.ask() and opt.tell() functions
    for trial_no in range(1, opt_iterations+1):
        [aug_type, magnitude] = opt.ask()
        [aug_type, magnitude] = [aug_type.tolist(), magnitude.tolist()]
        trial_hyperparams = [aug_type, magnitude]

        augmented_data = augmenter.run(data["X_train"], data["y_train"], aug_type, magnitude)

        sample_costs=[]
        for sample_no in range(1,opt_samples+1):
            child_model.load_pre_augment_weights()
            # TRAIN
            history = child_model.fit(data, augmented_data, epochs=child_epochs)
            #
            mean_late_val_acc = np.mean(history["val_acc"][-opt_last_n_epochs:])
            sample_costs.append(mean_late_val_acc)
            notebook.record(trial_no, trial_hyperparams, sample_no, mean_late_val_acc, history)

        trial_cost = np.mean(sample_costs)
        if trial_no%5==0:
            notebook.save()

        print(trial_no, trial_cost, trial_hyperparams)
        logging.info(f"{str(trial_no)}, {str(trial_cost)}, {str(trial_hyperparams)}")
        opt.tell(trial_hyperparams, trial_cost)

    notebook.save()
    print("End")

if __name__ == "__main__":

    run_bayesianopt()
