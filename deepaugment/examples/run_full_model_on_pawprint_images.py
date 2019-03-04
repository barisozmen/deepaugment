import numpy as np

import os
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
parent_dir_of_file = dirname(dir_of_file)
sys.path.insert(0, parent_dir_of_file)

from run_full_model import run_full_model


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


X, y = load_images("../../data/raw/pawprints/images")
# policies_path = "../../reports/experiments/pawprints_02-14_19-22/top20_policies.csv"
policies_path = "random"

run_full_model(X, y, epochs=200, batch_size=32, policies_path=policies_path)


