# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import numpy as np
from imgaug import augmenters as iaa


class Augmenter:
    """Augments given datasets
    """

    def run(self, X, y, aug_type, magnitude):
        """
        Args:
            X
            y
            params
        Returns:

        """
        X = (X * 255).copy()

        if aug_type == 1:
            X_aug = iaa.Crop(px=(0, int(magnitude * 32))).augment_images(X)
        elif aug_type == 2:
            X_aug = iaa.GaussianBlur(sigma=(0, magnitude * 1.0)).augment_images(X)
        elif aug_type == 3:
            X_aug = iaa.Affine(
                rotate=(-180 * magnitude, 180 * magnitude)
            ).augment_images(X)
        elif aug_type == 4:
            X_aug = iaa.Affine(shear=(-90 * magnitude, 90 * magnitude)).augment_images(
                X)
        elif aug_type == 5:
            X_aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.50, 5*magnitude)).augment_images(X)  # sharpen images
        elif aug_type == 6:
            X_aug = iaa.Emboss(alpha=(0, 1.0), strength=(0, 3.0*magnitude)).augment_images(X)  # emboss images
        elif aug_type == 7:
            X_aug = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, magnitude * 255), per_channel=0.5).augment_images(X)  # add gaussian noise to images
        elif aug_type == 8:
            X_aug = iaa.Dropout((0.01, magnitude), per_channel=0.5).augment_images(X)  # randomly remove up to 10% of the pixels
        elif aug_type == 9:
            X_aug = iaa.CoarseDropout((0.03, 0.15), size_percent=(0.30, np.log10(magnitude * 3)), per_channel=0.2).augment_images(X)
        else:
            raise AttributeError

        augmented_data = {"X_train": X_aug / 255, "y_train": y}
        return augmented_data
