# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

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
        X = (X*255).copy()

        if params[0] == 1:
            X_aug = iaa.Crop(px=(0,int(params[1]*32))).augment_images(X)
        elif params[0] == 2:
            X_aug = iaa.GaussianBlur(sigma=(0, params[1]*1.0)).augment_images(X)
        elif params[0] == 3:
            X_aug = iaa.Affine(rotate=(-180*params[1],180*params[1])).augment_images(X)
        elif params[0] == 4:
            X_aug = iaa.Affine(shear=(-90 * params[1], 90 * params[1])).augment_images(X)
        else:
            raise AttributeError

        augmented_data = {
            "X_train" : X_aug/255,
            "y_train" : y
        }
        return augmented_data
