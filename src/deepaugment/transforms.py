"""
Image transformations - elegant, minimal, powerful.

Uses torchvision v2 transforms. Convention: magnitude in [0, 1].
"""

from torchvision.transforms import v2 as T
from toolz import pipe
from .config import TRANSFORM_CATEGORIES
from .utils import numpy_to_pil, pil_to_numpy, clamp


# ============================================================
# TRANSFORM REGISTRY - Single Source of Truth
# ============================================================

TRANSFORMS = {
    # Geometric
    "rotate": lambda m: T.RandomRotation(degrees=int(180 * m)),
    "flip_h": lambda m: T.RandomHorizontalFlip(p=m),
    "flip_v": lambda m: T.RandomVerticalFlip(p=m),
    "affine": lambda m: T.RandomAffine(degrees=0, translate=(m * 0.2, m * 0.2)),
    "shear": lambda m: T.RandomAffine(degrees=0, shear=int(45 * m)),
    "perspective": lambda m: T.RandomPerspective(distortion_scale=0.5 * m, p=1.0),
    "elastic": lambda m: T.ElasticTransform(alpha=m * 50.0),
    "random_crop": lambda m: T.RandomResizedCrop(
        size=32, scale=(1 - m * 0.3, 1.0), ratio=(0.75, 1.33), antialias=True
    ),

    # Color
    "brightness": lambda m: T.ColorJitter(brightness=m * 0.5),
    "contrast": lambda m: T.ColorJitter(contrast=m * 0.5),
    "saturation": lambda m: T.ColorJitter(saturation=m * 0.5),
    "hue": lambda m: T.ColorJitter(hue=0.1 * m),
    "color_jitter": lambda m: T.ColorJitter(
        brightness=m * 0.3, contrast=m * 0.3, saturation=m * 0.3, hue=0.05 * m
    ),

    # Advanced color
    "sharpen": lambda m: T.RandomAdjustSharpness(sharpness_factor=1 + m * 3, p=1.0),
    "autocontrast": lambda m: T.RandomAutocontrast(p=1.0),
    "equalize": lambda m: T.RandomEqualize(p=1.0),
    "invert": lambda m: T.RandomInvert(p=1.0),
    "solarize": lambda m: T.RandomSolarize(threshold=int(128 + 127 * m), p=1.0),
    "posterize": lambda m: T.RandomPosterize(bits=max(1, int(2 + 6 * m)), p=1.0),
    "grayscale": lambda m: T.RandomGrayscale(p=m),

    # Blur
    "blur": lambda m: T.GaussianBlur(kernel_size=3, sigma=(0.1, 3 + 20 * m)),

    # Occlusion
    "erasing": lambda m: T.RandomErasing(p=m, scale=(0.02, 0.33), ratio=(0.3, 3.3)),

    # Advanced
    "channel_permute": lambda m: T.RandomChannelPermutation() if m > 0.5 else T.Identity(),
    "photometric_distort": lambda m: T.RandomPhotometricDistort(
        brightness=(1 - 0.3 * m, 1 + 0.3 * m),
        contrast=(1 - 0.3 * m, 1 + 0.3 * m),
        saturation=(1 - 0.3 * m, 1 + 0.3 * m),
        hue=(-0.05 * m, 0.05 * m),
    ),
}


# ============================================================
# TRANSFORM OPERATIONS - Functional, composable
# ============================================================

def make_transform(name, magnitude):
    """
    Create single transform. Fail fast if invalid.

    Convention: magnitude clamped to [0, 1].
    """
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {name}. Available: {list(TRANSFORMS.keys())}")

    mag = clamp(magnitude, 0.0, 1.0)
    return TRANSFORMS[name](mag)


def apply_policy(image, policy):
    """
    Apply augmentation policy to image.

    Pure function: image → policy → augmented image.

    Args:
        image: RGB image (H, W, C) numpy array or PIL Image
        policy: List of (transform_name, magnitude) tuples

    Returns:
        Augmented image as numpy array
    """
    # Pipeline: numpy → PIL → transforms → numpy
    was_numpy = hasattr(image, 'shape')  # Check if numpy
    pil_img = numpy_to_pil(image)

    # Build and apply transform pipeline
    transforms = [make_transform(name, mag) for name, mag in policy]
    pipeline = T.Compose(transforms)
    augmented = pipeline(pil_img)

    # Return in original format
    return pil_to_numpy(augmented) if was_numpy else augmented


def create_augmenter(policy):
    """
    Create reusable augmenter from policy.

    Returns: Callable that augments images.
    """
    transforms = [make_transform(name, mag) for name, mag in policy]
    pipeline = T.Compose(transforms)

    def augment(image):
        pil_img = numpy_to_pil(image)
        augmented = pipeline(pil_img)
        return pil_to_numpy(augmented)

    return augment


# ============================================================
# TRANSFORM CATEGORIES - Derived from config (SSOT)
# ============================================================

def get_transform_names(categories=None):
    """
    Get transform names, optionally filtered by category.

    Convention: None means all transforms.
    """
    if categories is None:
        return list(TRANSFORMS.keys())

    # Filter by categories
    allowed = []
    for cat in categories:
        if cat not in TRANSFORM_CATEGORIES:
            raise ValueError(f"Unknown category: {cat}. Available: {list(TRANSFORM_CATEGORIES.keys())}")
        allowed.extend(TRANSFORM_CATEGORIES[cat])

    return [t for t in TRANSFORMS.keys() if t in allowed]


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "TRANSFORMS",
    "make_transform",
    "apply_policy",
    "create_augmenter",
    "get_transform_names",
]
