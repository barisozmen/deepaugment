# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import numpy as np
from imgaug import augmenters as iaa


def normalize(X):
    return (X/255.).copy()

def denormalize(X):
    X_dn = (X*255).copy()
    X_dn = X_dn.astype(int)
    return X_dn

def transform(aug_type, magnitude, X):
    if aug_type == "crop":
        X_aug = iaa.Crop(px=(0, int(magnitude * 32))).augment_images(X)
    elif aug_type == "gaussian-blur":
        X_aug = iaa.GaussianBlur(sigma=(0, magnitude * 5.0)).augment_images(X)
    elif aug_type == "rotate":
        X_aug = iaa.Affine(rotate = (-180 * magnitude, 180 * magnitude)).augment_images(X)
    elif aug_type == "shear":
        X_aug = iaa.Affine(shear=(-90 * magnitude, 90 * magnitude)).augment_images(X)
    elif aug_type == "translate-x":
        X_aug = iaa.Affine(
            translate_percent={
                "x": (-magnitude, magnitude),
                "y": (0, 0)
            }
        ).augment_images(X)
    elif aug_type == "translate-y":
        X_aug = iaa.Affine(
            translate_percent={
                "x": (0, 0),
                "y": (-magnitude, magnitude)
            }
        ).augment_images(X)
    elif aug_type == "sharpen":
        X_aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.50, 5 * magnitude)).augment_images(X)
    elif aug_type == "emboss":
        X_aug = iaa.Emboss(alpha=(0, 1.0), strength=(0, 3.0 * magnitude)).augment_images(X)
    elif aug_type == "additive-gaussian-noise":
        X_aug = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, magnitude * 255), per_channel=0.5).augment_images(X)
    elif aug_type == "dropout":
        X_aug = iaa.Dropout((0.01, max(0.011, magnitude)), per_channel=0.5).augment_images(X) # Dropout first argument should be smaller than second one
    elif aug_type == "coarse-dropout":
        X_aug = iaa.CoarseDropout(
            (0.03, 0.15), size_percent=(0.30, np.log10(magnitude * 3)), per_channel=0.2
        ).augment_images(X)
    elif aug_type == "gamma-contrast":
        X_norm = normalize(X)
        X_aug_norm = iaa.GammaContrast(magnitude * 1.75).augment_images(X_norm) # needs 0-1 values
        X_aug = denormalize(X_aug_norm)
    elif aug_type == "brighten":
        X_aug = iaa.Add((int(-40 * magnitude), int(40 * magnitude)), per_channel=0.5).augment_images(X) # brighten
    elif aug_type == "invert":
        X_aug = iaa.Invert(1.0).augment_images(X) # magnitude not used
    elif aug_type == "fog":
        X_aug = iaa.Fog().augment_images(X) # magnitude not used
    elif aug_type == "clouds":
        X_aug = iaa.Clouds().augment_images(X) # magnitude not used
    elif aug_type == "histogram-equalize":
        X_aug = iaa.AllChannelsHistogramEqualization().augment_images(X) # magnitude not used
    elif aug_type == "super-pixels":
        X_aug = iaa.Superpixels(p_replace=(0, magnitude), n_segments=(100, 100)).augment_images(X)
    elif aug_type == "perspective-transform":
        X_aug = iaa.PerspectiveTransform(scale=(0.01, max(0.02, magnitude*0.125 ))).augment_images(X) # first scale param must be larger
    elif aug_type == "elastic-transform":
        X_aug = iaa.ElasticTransformation(alpha=(0.0, max(0.5, magnitude*60)), sigma=0.25).augment_images(X)
    elif aug_type == "add-to-hue-and-saturation":
        X_aug = iaa.AddToHueAndSaturation((int(-45*magnitude), int(45*magnitude))).augment_images(X)
    else:
        raise ValueError
    return X_aug

class Augmenter:
    """Augments given datasets
    """
    @staticmethod
    def run(X, y,
            aug1_type, aug1_magnitude,
            aug2_type, aug2_magnitude,
            portion
            ):
        """
        """
        assert portion>=0.0 and portion<=1.0, "portion argument value is out of accepted interval"

        # convert data to 255 from normalized
        _X = (X * 255).copy() # to 255
        _X = _X.astype(int)

        if portion==1.0:
            X_portion = _X
            y_portion = y
        else:
            # get a portion of data
            ix = np.random.choice(len(_X), int(len(_X)*portion), False)
            X_portion = _X[ix].copy()
            y_portion = y[ix].copy()

        # transform that portion
        X_portion_aug = transform(aug1_type, aug1_magnitude, X_portion) # first transform
        X_portion_aug = transform(aug2_type, aug2_magnitude, X_portion_aug) # second transform

        augmented_data = {"X_train": X_portion_aug / 255.0, "y_train": y_portion} # back to normalization

        return augmented_data # augmenteed data is mostly smaller than whole data










