import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
sys.path.insert(0, dir_of_file)

from deepaugment import DeepAugment
from build_features import DataOp

my_config = {
    "model": "basiccnn",
    "method": "bayesian_optimization",
    "train_set_size": 1000,
    "opt_samples": 3,
    "opt_last_n_epochs": 3,
    "opt_initial_points": 100,
    "child_epochs": 120,
    "child_first_train_epochs": 0,
    "child_batch_size": 64
}

deepaug = DeepAugment("cifar10", config=my_config)

best_policies = deepaug.optimize(3000)


#
# X, y, input_shape = DataOp.load("cifar10")
# train_size = int(len(X)*0.9)
#
# data = DataOp.preprocess(X, y, train_size)
#
#
# imgen = deepaug.image_generator_with_top_policies(data["X_train"], data["y_train"])

print("End")
