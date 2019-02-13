import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
sys.path.insert(0, dir_of_file)

from deepaugment import DeepAugment
from build_features import DataOp

deepaug = DeepAugment("cifar10", config={"child_epochs": 1})

best_policies = deepaug.optimize(1)



X, y, input_shape = DataOp.load("cifar10")


train_size = int(len(X)*0.9)

data = DataOp.preprocess(X, y, train_size)

imgen = deepaug.image_generator_with_top_policies(data["X_train"], data["y_train"])

print("End")
