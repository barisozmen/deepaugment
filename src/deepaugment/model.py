"""Lightweight CNN for evaluating augmentation policies."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


class SimpleCNN(nn.Module):
    """Minimal CNN for fast policy evaluation."""

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        # Compact but effective architecture
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 4x4
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def train_model(model, train_data, val_data, epochs=10, batch_size=64, lr=0.001, device=None):
    """
    Train model and return performance metrics.

    Args:
        model: PyTorch model
        train_data: (X_train, y_train) as numpy arrays
        val_data: (X_val, y_val) as numpy arrays
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda', 'mps', or 'cpu'

    Returns:
        Dictionary with training history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = model.to(device)

    # Prepare data
    X_train, y_train = train_data
    X_val, y_val = val_data

    # Convert to tensors (assume images are H,W,C format, convert to C,H,W)
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2) / 255.0
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2) / 255.0
    y_val = torch.LongTensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)

    return history


def evaluate_policy(policy, train_data, val_data, num_classes, augmenter, epochs=10, samples=3):
    """
    Evaluate augmentation policy by training model.

    Args:
        policy: List of (transform, magnitude) tuples
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        num_classes: Number of output classes
        augmenter: Function to augment images
        epochs: Training epochs
        samples: Number of times to train (average results)

    Returns:
        Average validation accuracy
    """
    scores = []

    for _ in range(samples):
        # Augment training data
        X_train, y_train = train_data
        X_aug = np.array([augmenter(img, policy) for img in X_train])

        # Train model
        model = SimpleCNN(num_classes=num_classes)
        history = train_model(
            model, (X_aug, y_train), val_data, epochs=epochs, batch_size=64
        )

        # Take best validation accuracy
        scores.append(max(history['val_acc']))

    return np.mean(scores)
