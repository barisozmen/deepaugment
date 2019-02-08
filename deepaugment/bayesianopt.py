# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import os
import sys
from os.path import dirname, realpath
file_path = realpath(__file__)
dir_of_file = dirname(file_path)
parent_dir_of_file = dirname(dir_of_file)
sys.path.insert(0, parent_dir_of_file)

# Set experiment name
import datetime

now = datetime.datetime.now()
EXPERIMENT_NAME = f"{now.year}-{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}"

import pandas as pd
import numpy as np
import skopt

# Import machine learning libraries
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # tell tensorflow not to use all resources
session = tf.Session(config=config)
import keras

keras.backend.set_session(session)

import pathlib
import logging

EXPERIMENT_FOLDER_PATH = os.path.join(parent_dir_of_file, f"reports/experiments/{EXPERIMENT_NAME}")
log_path = pathlib.Path(EXPERIMENT_FOLDER_PATH)
log_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=(log_path / "info.log").absolute(), level=logging.DEBUG)

# import modules from DeepAugmenter
from augmenter import Augmenter
from childcnn import ChildCNN
from notebook import Notebook
notebook = Notebook(f"{EXPERIMENT_FOLDER_PATH}/notebook.csv")
from build_features import DataOp
from lib.decorators import Reporter
logger = Reporter.logger

AUG_TYPES = [
    "crop", "gaussian-blur", "rotate", "shear", "translate-x", "translate-y", "sharpen",
    "emboss", "additive-gaussian-noise", "dropout", "coarse-dropout", "gamma-contrast",
    "brighten", "invert", "fog", "clouds", "add-to-hue-and-saturation", "coarse-salt-pepper",
    "horizontal-flip", "vertical-flip"
]



# warn user if TensorFlow does not see the GPU
from tensorflow.python.client import device_lib
if "GPU" not in str(device_lib.list_local_devices()):
    print("GPU not available!")
    logging.warning("GPU not available!")
# Note: GPU not among local devices means GPU not used for sure,
#       HOWEVER GPU among local devices does not guarantee it is used

def calculate_reward(history):
    history_df = pd.DataFrame(history)
    history_df["acc_overfit"] = history_df["acc"] - history_df["val_acc"]
    reward = (history_df[history_df["acc_overfit"]<=0.05]["val_acc"]
                .nlargest(3)
                .mean()
             )
    return reward

def objective(
    trial_no,
    data, child_model, augmenter,
    child_epochs, opt_samples, opt_last_n_epochs,
    trial_hyperparams
):
    augmented_data = augmenter.run(
        data["X_train"], data["y_train"],
        *trial_hyperparams
    )

    sample_rewards = []
    for sample_no in range(1, opt_samples + 1):
        child_model.load_pre_augment_weights()
        # TRAIN
        history = child_model.fit(data, augmented_data, epochs=child_epochs)
        #
        reward = calculate_reward(history)
        sample_rewards.append(reward)
        notebook.record(trial_no, trial_hyperparams, sample_no, reward, history)

    trial_cost = 1 - np.mean(sample_rewards)
    notebook.save()

    print(trial_no, trial_cost, trial_hyperparams)
    logging.info(f"{str(trial_no)}, {str(trial_cost)}, {str(trial_hyperparams)}")

    return trial_cost


import click
@click.command()
@click.option("--dataset-name", type=click.STRING, default="cifar10")
@click.option("--model-name", type=click.STRING, default="wrn_40_4")
@click.option("--num-classes", type=click.INT, default=10)
@click.option("--train-set-size", type=click.INT, default=4000)
@click.option("--opt-iterations", type=click.INT, default=1000)
@click.option("--opt-samples", type=click.INT, default=5)
@click.option("--opt-last-n-epochs", type=click.INT, default=5)
@click.option("--opt-initial-points", type=click.INT, default=20)
@click.option("--child-epochs", type=click.INT, default=15)
@click.option("--child-first-train-epochs", type=click.INT, default=0)
@click.option("--child-batch-size", type=click.INT, default=32)
@logger(logfile_dir=EXPERIMENT_FOLDER_PATH)
def run_bayesianopt(
    dataset_name,
    model_name,
    num_classes,
    train_set_size,
    opt_iterations,
    opt_samples,
    opt_last_n_epochs,
    opt_initial_points,
    child_epochs,
    child_first_train_epochs,
    child_batch_size,
):
    data, input_shape = DataOp.load(dataset_name, train_set_size)
    data = DataOp.preprocess(data)

    child_model = ChildCNN(
        model_name, input_shape, child_batch_size, num_classes,
        "initial_model_weights.h5", logging
    )
    # first training
    if child_first_train_epochs>0:
        history = child_model.fit(data, epochs=child_first_train_epochs)
        notebook.record(0, ["first", 0.0,"first",0.0,"first",0.0,0.0], 1, None, history)
    #
    child_model.model.save_weights(child_model.pre_augmentation_weights_path)
    augmenter = Augmenter()

    ####################################################################################################
    # Implementation of skopt by ask-tell design pattern
    # See https://geekyisawesome.blogspot.com/2018/07/hyperparameter-tuning-using-scikit.html
    ####################################################################################################
    opt = skopt.Optimizer(
        [
            skopt.space.Categorical(AUG_TYPES, name='aug1_type'),
            skopt.space.Real(0.0, 1.0, name='aug1_magnitude'),
            skopt.space.Categorical(AUG_TYPES, name='aug2_type'),
            skopt.space.Real(0.0, 1.0, name='aug2_magnitude'),
            skopt.space.Real(0.0, 1.0, name='portion')
        ],
        n_initial_points=opt_initial_points,
        base_estimator='RF', # Random Forest estimator
        acq_func='EI', # Expected Improvement
        acq_optimizer='auto',
        random_state=0
    )
    # skopt works with opt.ask() and opt.tell() functions
    for trial_no in range(1, opt_iterations+1):
        trial_hyperparams = opt.ask()
        print("trial:",trial_no)
        print(trial_hyperparams)

        f_val = objective(
            trial_no=trial_no,
            data=data, child_model=child_model, augmenter=augmenter,
            child_epochs=child_epochs, opt_samples=opt_samples, opt_last_n_epochs=opt_last_n_epochs,
            trial_hyperparams=trial_hyperparams
        )
        opt.tell(trial_hyperparams, f_val)

    # run without augmentation
    objective(
        trial_no=trial_no+1,
        data=data, child_model=child_model, augmenter=augmenter,
        child_epochs=child_epochs, opt_samples=opt_samples, opt_last_n_epochs=opt_last_n_epochs,
        trial_hyperparams=["crop",0.0,"crop",0.0,0.0]
    )

    # get top-20 policies
    top20_df = notebook.get_top_policies(k=20)
    notebook.save()




    print("End")

if __name__ == "__main__":

    run_bayesianopt()
