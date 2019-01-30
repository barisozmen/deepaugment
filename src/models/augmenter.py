# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import numpy as np
from imgaug import augmenters as iaa


class Augmenter:
    """Augments given datasets
    """

    def run(self, X, y, params):
        """
        Args:
            X
            y
            params
        Returns:

        """
        X = (X * 255).copy()

        if params[0] == 1:
            X_aug = iaa.Crop(px=(0, int(params[1] * 32))).augment_images(X)
        elif params[0] == 2:
            X_aug = iaa.GaussianBlur(sigma=(0, params[1] * 1.0)).augment_images(X)
        elif params[0] == 3:
            X_aug = iaa.Affine(
                rotate=(-180 * params[1], 180 * params[1])
            ).augment_images(X)
        elif params[0] == 4:
            X_aug = iaa.Affine(shear=(-90 * params[1], 90 * params[1])).augment_images(
                X)
        elif params[0] == 5:
            X_aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.50, 5*params[1])).augment_images(X)  # sharpen images
        elif params[0] == 6:
            X_aug = iaa.Emboss(alpha=(0, 1.0), strength=(0, 3.0*params[1])).augment_images(X)  # emboss images
        elif params[0] == 7:
            X_aug = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, params[1] * 255), per_channel=0.5).augment_images(X)  # add gaussian noise to images
        elif params[0] == 8:
            X_aug = iaa.Dropout((0.01, params[1]), per_channel=0.5).augment_images(X)  # randomly remove up to 10% of the pixels
        elif params[0] == 9:
            X_aug = iaa.CoarseDropout((0.03, 0.15), size_percent=(0.30, np.log10(params[1] * 3)), per_channel=0.2).augment_images(X)
        else:
            raise AttributeError

        augmented_data = {"X_train": X_aug / 255, "y_train": y}
        return augmented_data
