"""
Elegant utilities inspired by Unix philosophy and functional programming.

Each function does one thing well. Compose them for power.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from toolz import pipe, curry


# ============================================================
# IMAGE CONVERSIONS - Functional and composable
# ============================================================

def numpy_to_pil(image):
    """numpy → PIL. Pure function, no side effects."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    return image


def pil_to_numpy(image):
    """PIL → numpy. Pure function, no side effects."""
    if isinstance(image, Image.Image):
        return np.array(image)
    return image


# ============================================================
# DATA SAMPLING - Convention: reproducible by default
# ============================================================

@curry
def sample_indices(total, size, seed=42):
    """Sample indices without replacement. Functional and reproducible."""
    np.random.seed(seed)
    return np.random.choice(total, min(size, total), replace=False)


def split_data(X, y, ratio=0.8, seed=42):
    """Split data into train/val. Convention: 80/20 split."""
    split_idx = int(ratio * len(X))
    return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:])


# ============================================================
# NUMPY/JSON SERIALIZATION - Handle all numpy types elegantly
# ============================================================

def to_native(obj):
    """
    Convert numpy types to native Python for JSON serialization.

    Recursive, handles nested structures. Unix philosophy: do one thing well.
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [to_native(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_native(val) for key, val in obj.items()}
    return obj


# ============================================================
# CHECKPOINTING - Simple, elegant persistence
# ============================================================

def save_checkpoint(data, filename, directory=None):
    """Save checkpoint as JSON. Convention: pretty print for human readability."""
    if directory:
        Path(directory).mkdir(exist_ok=True)
        filepath = Path(directory) / filename
    else:
        filepath = Path(filename)

    with open(filepath, "w") as f:
        json.dump(to_native(data), f, indent=2)

    return filepath


def load_checkpoint(filepath):
    """Load checkpoint from JSON. Simple, no magic."""
    with open(filepath) as f:
        return json.load(f)


# ============================================================
# EXPERIMENT NAMING - Convention: timestamp-based
# ============================================================

def generate_experiment_name(prefix="exp"):
    """Generate unique experiment name. Convention: ISO format timestamp."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================
# POLICY FORMATTING - Beautiful output
# ============================================================

def format_policy(policy, indent=4):
    """Format policy as beautiful string for display."""
    lines = []
    for transform, magnitude in policy:
        lines.append(f"{' ' * indent}{transform:20s} magnitude={magnitude:.3f}")
    return "\n".join(lines)


def format_policy_summary(policy):
    """One-line policy summary."""
    return ", ".join(f"{t}({m:.2f})" for t, m in policy)


# ============================================================
# VALIDATION - Fail fast with clear messages
# ============================================================

def validate_images(X):
    """Validate image array format. Fail fast with helpful error."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Images must be numpy array, got {type(X)}")

    if X.ndim != 4:
        raise ValueError(f"Images must be 4D (N, H, W, C), got shape {X.shape}")

    if X.shape[-1] not in [1, 3]:
        raise ValueError(f"Images must have 1 or 3 channels, got {X.shape[-1]}")

    return True


def validate_labels(y, X):
    """Validate labels match images. Fail fast."""
    if len(y) != len(X):
        raise ValueError(f"Labels ({len(y)}) don't match images ({len(X)})")
    return True


# ============================================================
# FUNCTIONAL HELPERS - Compose elegantly
# ============================================================

def clamp(x, min_val=0.0, max_val=1.0):
    """Clamp value to range. Simple, pure."""
    return np.clip(x, min_val, max_val)


def normalize(arr, low=0.0, high=1.0):
    """Normalize array to range [low, high]."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.full_like(arr, (high + low) / 2)
    return low + (arr - arr_min) * (high - low) / (arr_max - arr_min)
