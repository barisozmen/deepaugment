
import keras

import numpy as np
import pandas as pd

from augmenter import Augmenter
from lib.cutout import cutout_numpy


def random_flip(x):
  """Flip the input x horizontally with 50% probability."""
  if np.random.rand(1)[0] > 0.5:
    return np.fliplr(x)
  return x

def zero_pad_and_crop(img, amount=4):
  """Zero pad by `amount` zero pixels on each side then take a random crop.
  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.
  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
  padded_img = np.zeros((img.shape[0] + amount * 2, img.shape[1] + amount * 2,
                         img.shape[2]))
  padded_img[amount:img.shape[0] + amount, amount:
             img.shape[1] + amount, :] = img
  top = np.random.randint(low=0, high=2 * amount)
  left = np.random.randint(low=0, high=2 * amount)
  new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
  return new_img



def apply_default_transformations(X):
    # apply cutout
    X_aug = []
    for img in X:
        img_aug = zero_pad_and_crop(img, amount=4)
        img_aug = cutout_numpy(img_aug, size=6)
        X_aug.append(img_aug)
    return X_aug


def deepaugment_image_generator(X, y, policy, batch_size=64, augment_chance=0.5):
    """Yields batch of images after applying random augmentations from the policy

    Each image is augmented by 50% chance. If augmented, one of the augment-chain in the policy is applied.
    Which augment-chain to apply is chosen randomly.

    Args:
        X (numpy.array):
        labels (numpy.array):
        policy (list): list of dictionaries

    Returns:
    """
    if type(policy)==str:
        policy_df = pd.read_csv(policy)
        policy_df = policy_df[["aug1_type","aug1_magnitude","aug2_type","aug2_magnitude","portion"]]
        policy = policy_df.to_dict(orient="records")

    print("Policies are:")
    print(policy)
    print()

    augmenter = Augmenter()

    while True:
        ix = np.arange(len(X))
        np.random.shuffle(ix)
        for i in range(len(X) // batch_size):
            _ix = ix[i * batch_size:(i + 1) * batch_size]
            _X = X[_ix]
            _y = y[_ix]

            tiny_batch_size=4
            aug_X = _X[0:tiny_batch_size]
            aug_y = _y[0:tiny_batch_size]
            for j in range(1, len(_X) // tiny_batch_size):
                tiny_X = _X[j * tiny_batch_size:(j + 1) * tiny_batch_size]
                tiny_y = _y[j * tiny_batch_size:(j + 1) * tiny_batch_size]
                if np.random.rand() <= augment_chance:
                    aug_chain = np.random.choice(policy)
                    aug_chain['portion'] = 1.0 # last element is portion, which we want to be 1
                    hyperparams = list(aug_chain.values())

                    aug_data = augmenter.run(tiny_X, tiny_y, *hyperparams)

                    aug_data["X_train"] = apply_default_transformations(aug_data["X_train"])

                    aug_X = np.concatenate([aug_X, aug_data["X_train"]])
                    aug_y = np.concatenate([aug_y, aug_data["y_train"]])
                else:
                    aug_X = np.concatenate([aug_X, tiny_X])
                    aug_y = np.concatenate([aug_y, tiny_y])
            yield aug_X, aug_y


def test_deepaugment_image_generator():
    X = np.random.rand(200, 32, 32, 3)

    y = np.random.randint(10, size=200)
    y = keras.utils.to_categorical(y)

    batch_size=64

    policy = [
        {
            "aug1_type": "sharpen",
            "aug1_magnitude": 0.5,
            "aug2_type": "rotate",
            "aug2_magnitude": 0.2,
            "aug3_type": "emboss",
            "aug3_magnitude": 0.2,
            "portion": 0.5,
        },
        {
            "aug1_type": "gamma-contrast",
            "aug1_magnitude": 0.5,
            "aug2_type": "dropout",
            "aug2_magnitude": 0.2,
            "aug3_type": "clouds",
            "aug3_magnitude": 0.2,
            "portion": 0.2,
        }
    ]

    return deepaugment_generator(X,y,policy,batch_size=batch_size)



if __name__ == "__main__":
    gen = test_deepaugment_generator()

    a = next(gen)
    b = next(gen)
    c = next(gen)

    pass