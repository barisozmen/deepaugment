
import keras

import numpy as np

from augmenter import Augmenter


def deepaugment_generator(X, y, policy, batch_size=64, augment_chance=0.5):
    """Yields batch of images after applying random augmentations from the policy

    Each image is augmented by 50% chance. If augmented, one of the augment-chain in the policy is applied.
    Which augment-chain to apply is chosen randomly.

    Args:
        X (numpy.array):
        labels (numpy.array):
        policy (list): list of dictionaries

    Returns:
    """
    augmenter = Augmenter

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
                if np.random.rand() < augment_chance:
                    aug_chain = np.random.choice(policy)
                    aug_chain['portion'] = 1.0 # last element is portion, which we want to be 1
                    hyperparams = list(aug_chain.values())
                    aug_data = augmenter.run(tiny_X, tiny_y, *hyperparams)
                    aug_X = np.concatenate([aug_X, aug_data["X_train"]])
                    aug_y = np.concatenate([aug_y, aug_data["y_train"]])
                else:
                    aug_X = np.concatenate([aug_X, tiny_X])
                    aug_y = np.concatenate([aug_y, tiny_y])
            yield aug_X, aug_y


def test_deepaugment_generator():
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