"""
Model architectures - minimal, elegant, effective.

Just the models. Training logic lives in trainer.py (separation of concerns).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MODEL_CONFIGS


# ============================================================
# SIMPLE CNN - Fast evaluation model
# ============================================================

class SimpleCNN(nn.Module):
    """
    Minimal CNN for fast policy evaluation.

    Compact but effective: ~1.2M parameters for 32x32 images.
    """

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()

        # Get config from SSOT
        cfg = MODEL_CONFIGS["simple"]
        c1, c2, c3 = cfg["channels"]
        fc_size = cfg["fc_size"]
        dropout = cfg["dropout"]

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c3 * 4 * 4, fc_size)  # After 3 poolings: 32→16→8→4
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        """Forward pass. Clean and simple."""
        x = self.pool(F.relu(self.conv1(x)))  # 32 → 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 → 8
        x = self.pool(F.relu(self.conv3(x)))  # 8 → 4
        x = x.view(x.size(0), -1)             # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ============================================================
# MODEL FACTORY - Convention over Configuration
# ============================================================

def create_model(model_name="simple", num_classes=10, in_channels=3):
    """
    Create model by name. Convention: 'simple' is default.

    Extensible: add more models here as needed.
    """
    if model_name == "simple":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: ['simple']")
