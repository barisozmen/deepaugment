import numpy as np

import os
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../")

from deepaugment.deepaugment import DeepAugment


def load_images(image_dir_path):

    subfolders = next(os.walk(image_dir_path))[1]

    img_class = 0
    X_list = []
    y_list = []

    for subfolder in subfolders:

        subfolder_path = os.path.join(image_dir_path, subfolder)
        print(subfolder_path)

        for f in os.listdir(subfolder_path):

            if f.startswith("."): # dont look .DS_store
                print (f)
                continue

            img = image.load_img(os.path.join(subfolder_path,f), target_size=(100, 100))
            img_arr = image.img_to_array(img)
            X_list.append(img_arr)
            y_list.append(img_class)

        img_class+=1

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


X, y = load_images("../../data/images")

my_config = {
    "model": "basiccnn",
    "method": "bayesian_optimization",
    "train_set_size": 1000,
    "opt_samples": 3,
    "opt_last_n_epochs": 3,
    "opt_initial_points": 10,
    "child_epochs": 25,
    "child_first_train_epochs": 0,
    "child_batch_size": 64
}

deepaug = DeepAugment(images=X, labels=y, config=my_config)

best_policies = deepaug.optimize(500)

print (best_policies)