"""
Single Source of Truth for all configuration.

Following Rails doctrine: Convention over Configuration.
All magic numbers live here. Change once, apply everywhere.
"""

from pathlib import Path
from attrs import define, field

# ============================================================
# PATHS - Single source of truth for all file locations
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist (Convention: create if needed)
for d in [DATA_DIR, EXPERIMENTS_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True)


# ============================================================
# DEFAULTS - Convention over Configuration
# ============================================================

@define
class Defaults:
    """Beautiful defaults that just work."""

    # Training
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001

    # Optimization
    iterations: int = 50
    samples: int = 1  # Training runs per policy
    n_operations: int = 4  # Transforms per policy
    n_initial_points: int = 10  # Random exploration first

    # Data
    train_size: int = 2000
    val_size: int = 500
    image_size: int = 32  # Standard CIFAR-10 size

    # Model
    model_name: str = "simple"
    dropout_rate: float = 0.3

    # Device
    device: str = "auto"  # cuda > mps > cpu

    # Search
    method: str = "bayesian"  # or "random"
    random_state: int = 42

    # Persistence
    save_history: bool = True
    checkpoint_every: int = 10  # Save every N iterations

    # Early stopping
    early_stopping: bool = False
    patience: int = 10


# Singleton instance - The Source of Truth
defaults = Defaults()


# ============================================================
# TRANSFORM CATEGORIES - Derived, not duplicated
# ============================================================

TRANSFORM_CATEGORIES = {
    "geometric": [
        "rotate", "flip_h", "flip_v", "affine",
        "shear", "perspective", "elastic", "random_crop"
    ],
    "color": [
        "brightness", "contrast", "saturation", "hue", "color_jitter"
    ],
    "advanced_color": [
        "sharpen", "autocontrast", "equalize", "invert",
        "solarize", "posterize", "grayscale"
    ],
    "blur_noise": ["blur"],
    "occlusion": ["erasing"],
    "advanced": ["channel_permute", "photometric_distort"],
}

# All transforms (derived from categories)
ALL_TRANSFORMS = [t for cats in TRANSFORM_CATEGORIES.values() for t in cats]


# ============================================================
# MODEL CONFIGS - Architecture specifications
# ============================================================

MODEL_CONFIGS = {
    "simple": {
        "channels": [32, 64, 128],
        "fc_size": 256,
        "dropout": defaults.dropout_rate,
    },
}


# ============================================================
# HELPERS - Convention: smart auto-detection
# ============================================================

def auto_device():
    """Auto-detect best available device. CUDA > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device):
    """Resolve device string. Convention: 'auto' means smart detection."""
    return auto_device() if device == "auto" else device
