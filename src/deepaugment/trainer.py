"""
Training logic - functional, composable, minimal.

Pure functions for training models. No classes, just functions.
Unix philosophy: do one thing well.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from toolz import pipe
from .config import defaults, resolve_device


# ============================================================
# DATA PREPARATION - Pure functions
# ============================================================

def prepare_data(X, y):
    """
    Convert numpy to torch tensors.

    Convention: assumes (N, H, W, C) format, converts to (N, C, H, W).
    """
    X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2) / 255.0
    y_tensor = torch.LongTensor(y)
    return X_tensor, y_tensor


def create_loaders(train_data, val_data, batch_size=None):
    """
    Create data loaders. Convention: sensible batch size from config.
    """
    batch_size = batch_size or defaults.batch_size

    X_train, y_train = prepare_data(*train_data)
    X_val, y_val = prepare_data(*val_data)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size
    )

    return train_loader, val_loader


# ============================================================
# TRAINING - One epoch
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train for one epoch. Pure function, returns metrics.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
    }


# ============================================================
# VALIDATION - One epoch
# ============================================================

def validate_epoch(model, loader, criterion, device):
    """
    Validate for one epoch. Pure function, returns metrics.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
    }


# ============================================================
# FULL TRAINING - Compose the pieces
# ============================================================

def train_model(
    model,
    train_data,
    val_data,
    epochs=None,
    batch_size=None,
    learning_rate=None,
    device=None,
):
    """
    Train model to completion. Functional composition.

    Args:
        model: PyTorch model
        train_data: (X_train, y_train) as numpy arrays
        val_data: (X_val, y_val) as numpy arrays
        epochs: Training epochs (default from config)
        batch_size: Batch size (default from config)
        learning_rate: Learning rate (default from config)
        device: Device string (default from config)

    Returns:
        Training history dict
    """
    # Convention: use defaults from config
    epochs = epochs or defaults.epochs
    batch_size = batch_size or defaults.batch_size
    learning_rate = learning_rate or defaults.learning_rate
    device = resolve_device(device or defaults.device)

    # Setup
    model = model.to(device)
    train_loader, val_loader = create_loaders(train_data, val_data, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

    return history


# ============================================================
# POLICY EVALUATION - High-level interface
# ============================================================

def evaluate_policy(
    policy,
    train_data,
    val_data,
    num_classes,
    augmenter,
    model_factory,
    epochs=None,
    samples=None,
    **kwargs
):
    """
    Evaluate augmentation policy by training model.

    Functional approach: augment → train → evaluate → score.

    Args:
        policy: List of (transform, magnitude) tuples
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        num_classes: Number of classes
        augmenter: Function to augment images
        model_factory: Function that creates model
        epochs: Training epochs
        samples: Number of training runs (for averaging)
        **kwargs: Additional args for train_model

    Returns:
        Average validation accuracy
    """
    samples = samples or defaults.samples
    epochs = epochs or defaults.epochs

    scores = []

    for _ in range(samples):
        # Augment training data
        X_train, y_train = train_data
        X_aug = np.array([augmenter(img, policy) for img in X_train])

        # Train fresh model
        model = model_factory(num_classes=num_classes)
        history = train_model(model, (X_aug, y_train), val_data, epochs=epochs, **kwargs)

        # Take best validation accuracy
        scores.append(max(history["val_acc"]))

    return np.mean(scores)
