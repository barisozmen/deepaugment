# (C) 2019 Baris Ozmen <hbaristr@gmail.com>
# (C) 2024 Peter Norvig

import numpy as np
from imgaug import augmenters as iaa

def _normalize(images):
    return (images / 255.0).copy()

def _denormalize(images):
    return (images * 255).copy()

AUGMENTER_MAP = {
    "crop": lambda m, i: iaa.Crop(px=(0, int(m * 32))).augment_images(i),
    "gaussian-blur": lambda m, i: iaa.GaussianBlur(sigma=(0, m * 25.0)).augment_images(i),
    "rotate": lambda m, i: iaa.Affine(rotate=(-180 * m, 180 * m)).augment_images(i),
    "shear": lambda m, i: iaa.Affine(shear=(-90 * m, 90 * m)).augment_images(i),
    "translate-x": lambda m, i: iaa.Affine(translate_percent={"x": (-m, m), "y": (0, 0)}).augment_images(i),
    "translate-y": lambda m, i: iaa.Affine(translate_percent={"x": (0, 0), "y": (-m, m)}).augment_images(i),
    "horizontal-flip": lambda m, i: iaa.Fliplr(m).augment_images(i),
    "vertical-flip": lambda m, i: iaa.Flipud(m).augment_images(i),
    "sharpen": lambda m, i: iaa.Sharpen(alpha=(0, 1.0), lightness=(0.50, 5 * m)).augment_images(i),
    "emboss": lambda m, i: iaa.Emboss(alpha=(0, 1.0), strength=(0.0, 20.0 * m)).augment_images(i),
    "additive-gaussian-noise": lambda m, i: iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, m * 255), per_channel=0.5).augment_images(i),
    "dropout": lambda m, i: iaa.Dropout((0.01, max(0.011, m)), per_channel=0.5).augment_images(i),
    "coarse-dropout": lambda m, i: iaa.CoarseDropout((0.03, 0.15), size_percent=(0.30, np.log10(m * 3)), per_channel=0.2).augment_images(i),
    "gamma-contrast": lambda m, i: _denormalize(iaa.GammaContrast(m * 1.75).augment_images(_normalize(i))),
    "brighten": lambda m, i: iaa.Add((int(-40 * m), int(40 * m)), per_channel=0.5).augment_images(i),
    "invert": lambda m, i: iaa.Invert(1.0).augment_images(i),
    "fog": lambda m, i: iaa.Fog().augment_images(i),
    "clouds": lambda m, i: iaa.Clouds().augment_images(i),
    "histogram-equalize": lambda m, i: iaa.AllChannelsHistogramEqualization().augment_images(i),
    "add-to-hue-and-saturation": lambda m, i: iaa.AddToHueAndSaturation((int(-45 * m), int(45 * m))).augment_images(i),
    "coarse-salt-pepper": lambda m, i: iaa.CoarseSaltAndPepper(p=0.2, size_percent=m).augment_images(i),
    "grayscale": lambda m, i: iaa.Grayscale(alpha=(0.0, m)).augment_images(i),
}

def transform(aug_type, magnitude, images):
    if aug_type not in AUGMENTER_MAP:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
    return AUGMENTER_MAP[aug_type](magnitude, images)

def augment_by_policy(X, y, *hyperparams):
    images = _denormalize(X)

    augmented_images = []
    augmented_labels = []

    for i in range(0, len(hyperparams), 4):
        policy_images = images.copy()

        policy_images = transform(hyperparams[i], hyperparams[i+1], policy_images)
        np.clip(policy_images, 0, 255, out=policy_images)

        policy_images = transform(hyperparams[i+2], hyperparams[i+3], policy_images)
        np.clip(policy_images, 0, 255, out=policy_images)

        augmented_images.append(policy_images)
        augmented_labels.append(y)

    return {
        "X_train": _normalize(np.concatenate(augmented_images)),
        "y_train": np.concatenate(augmented_labels),
    }
