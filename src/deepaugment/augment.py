"""Image augmentation transformations using torchvision v2."""

import numpy as np
from torchvision.transforms import v2 as T
from PIL import Image


# transform registry - comprehensive torchvision v2 transforms
# see https://github.com/pytorch/vision/blob/218d2ab791d437309f91e0486eb9fa7f00badc17/torchvision/transforms/transforms.py
TRANSFORMS = {
    # Geometric transforms
    "rotate": lambda m: T.RandomRotation(degrees=int(180 * m)),
    "flip_h": lambda m: T.RandomHorizontalFlip(p=m),
    "flip_v": lambda m: T.RandomVerticalFlip(p=m),
    "affine": lambda m: T.RandomAffine(degrees=0, translate=(m * 0.2, m * 0.2)),
    "shear": lambda m: T.RandomAffine(degrees=0, shear=int(45 * m)),
    "perspective": lambda m: T.RandomPerspective(distortion_scale=0.5 * m, p=1.0),
    "elastic": lambda m: T.ElasticTransform(alpha=m * 50.0),
    # Color/photometric transforms
    "brightness": lambda m: T.ColorJitter(brightness=m * 0.5),
    "contrast": lambda m: T.ColorJitter(contrast=m * 0.5),
    "saturation": lambda m: T.ColorJitter(saturation=m * 0.5),
    "hue": lambda m: T.ColorJitter(hue=0.1 * m),
    "color_jitter": lambda m: T.ColorJitter(
        brightness=m * 0.3, contrast=m * 0.3, saturation=m * 0.3, hue=0.05 * m
    ),
    # Advanced color transforms
    "sharpen": lambda m: T.RandomAdjustSharpness(sharpness_factor=1 + m * 3, p=1.0),
    "autocontrast": lambda m: T.RandomAutocontrast(p=1.0),
    "equalize": lambda m: T.RandomEqualize(p=1.0),
    "invert": lambda m: T.RandomInvert(p=1.0),
    "solarize": lambda m: T.RandomSolarize(threshold=int(128 + 127 * m), p=1.0),
    "posterize": lambda m: T.RandomPosterize(bits=max(1, int(2 + 6 * m)), p=1.0),
    "grayscale": lambda m: T.RandomGrayscale(p=m),
    # Blur and noise
    "blur": lambda m: T.GaussianBlur(kernel_size=3, sigma=(0.1, 3 + 20 * m)),
    # Occlusion/masking
    "erasing": lambda m: T.RandomErasing(p=m, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    # 'cutout': lambda m: T.RandomErasing(p=1.0, scale=(0.02, m*0.4), ratio=(1.0, 1.0)),
    # Advanced augmentation techniques
    "channel_permute": lambda m: T.RandomChannelPermutation()
    if m > 0.5
    else T.Identity(),
    "photometric_distort": lambda m: T.RandomPhotometricDistort(
        brightness=(1 - 0.3 * m, 1 + 0.3 * m),
        contrast=(1 - 0.3 * m, 1 + 0.3 * m),
        saturation=(1 - 0.3 * m, 1 + 0.3 * m),
        hue=(-0.05 * m, 0.05 * m),
    ),
    # Crop and resize (useful for scale invariance)
    "random_crop": lambda m: T.RandomResizedCrop(
        size=32,  # Will be adjusted based on input size
        scale=(1 - m * 0.3, 1.0),
        ratio=(0.75, 1.33),
        antialias=True,
    ),
}


def get_transform_categories():
    """Get transforms organized by category."""
    return {
        "geometric": [
            "rotate",
            "flip_h",
            "flip_v",
            "affine",
            "shear",
            "perspective",
            "elastic",
            "random_crop",
        ],
        "color": ["brightness", "contrast", "saturation", "hue", "color_jitter"],
        "advanced_color": [
            "sharpen",
            "autocontrast",
            "equalize",
            "invert",
            "solarize",
            "posterize",
            "grayscale",
        ],
        "blur_noise": ["blur"],
        "occlusion": ["erasing"],
        "advanced": ["channel_permute", "photometric_distort"],
    }


def numpy_to_pil(image):
    """Convert numpy array to PIL Image."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    return image


def pil_to_numpy(image):
    """Convert PIL Image to numpy array."""
    if isinstance(image, Image.Image):
        return np.array(image)
    return image


def make_transform(name, magnitude):
    """Create a single transform with given magnitude (0-1 scale)."""
    if name not in TRANSFORMS:
        raise ValueError(
            f"Unknown transform: {name}. Available: {list(TRANSFORMS.keys())}"
        )
    return TRANSFORMS[name](np.clip(magnitude, 0.0, 1.0))


def apply_policy(image, policy):
    """
    Apply augmentation policy to an image.

    Args:
        image: RGB image as numpy array (H, W, C) or PIL Image
        policy: List of (transform_name, magnitude) tuples

    Returns:
        Augmented image as numpy array
    """
    # Convert to PIL if needed
    was_numpy = isinstance(image, np.ndarray)
    pil_image = numpy_to_pil(image)

    # Get original size for size-dependent transforms
    orig_size = pil_image.size  # (W, H)

    # Apply transforms
    transforms = []
    for name, mag in policy:
        transform = make_transform(name, mag)

        # Adjust size for crop transforms
        if name == "random_crop":
            transform = T.RandomResizedCrop(
                size=min(orig_size),
                scale=(1 - mag * 0.3, 1.0),
                ratio=(0.75, 1.33),
                antialias=True,
            )

        transforms.append(transform)

    pipeline = T.Compose(transforms)
    augmented = pipeline(pil_image)

    # Convert back to numpy if input was numpy
    if was_numpy:
        return pil_to_numpy(augmented)
    return augmented


def create_augmenter(policy):
    """Create reusable augmentation pipeline from policy."""
    transforms = [make_transform(name, mag) for name, mag in policy]
    return T.Compose(transforms)
