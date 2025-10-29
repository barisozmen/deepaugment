Custom Models
=============

Learn how to use your own models with DeepAugment.

Basic Custom Model
------------------

Any PyTorch ``nn.Module`` can be used:

.. code-block:: python

   import torch.nn as nn
   from deepaugment import DeepAugment

   class MyModel(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.features = nn.Sequential(
               nn.Conv2d(3, 32, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Conv2d(32, 64, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
           )
           self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(64 * 8 * 8, 128),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(128, num_classes),
           )

       def forward(self, x):
           x = self.features(x)
           x = self.classifier(x)
           return x

   # Use your model
   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       model=MyModel  # Pass the class, not an instance
   )

   best_policy = aug.optimize(iterations=50)

**Important**: Pass the model **class**, not an instance. DeepAugment will instantiate it internally with the correct ``num_classes``.

ResNet Example
--------------

Using torchvision's ResNet:

.. code-block:: python

   from torchvision.models import resnet18
   import torch.nn as nn
   from deepaugment import DeepAugment

   class ResNet18Wrapper(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.model = resnet18(weights=None)
           # Modify first conv for 32x32 images
           self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
           self.model.maxpool = nn.Identity()  # Remove maxpool for small images
           # Adjust final layer
           self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

       def forward(self, x):
           return self.model(x)

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       model=ResNet18Wrapper,
       train_size=1000,  # Smaller subset for larger model
       val_size=200
   )

   best_policy = aug.optimize(iterations=50, epochs=15)

EfficientNet Example
--------------------

Using EfficientNet from timm:

.. code-block:: python

   import timm
   import torch.nn as nn
   from deepaugment import DeepAugment

   class EfficientNetWrapper(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)

       def forward(self, x):
           return self.model(x)

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       model=EfficientNetWrapper,
       batch_size=32,  # Smaller batch for memory
   )

   best_policy = aug.optimize(iterations=50)

Vision Transformer Example
---------------------------

Using ViT:

.. code-block:: python

   import timm
   import torch.nn as nn
   from deepaugment import DeepAugment

   class ViTWrapper(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.model = timm.create_model(
               'vit_tiny_patch16_224',
               pretrained=False,
               num_classes=num_classes,
               img_size=32  # Adjust for CIFAR-10
           )

       def forward(self, x):
           return self.model(x)

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       model=ViTWrapper,
       train_size=500,
       batch_size=16,
       learning_rate=0.001,
   )

   best_policy = aug.optimize(iterations=50, epochs=20)

Model with Batch Normalization
-------------------------------

Models with BatchNorm work seamlessly:

.. code-block:: python

   import torch.nn as nn
   from deepaugment import DeepAugment

   class ConvNetBN(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
           self.bn1 = nn.BatchNorm2d(64)
           self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
           self.bn2 = nn.BatchNorm2d(128)
           self.fc = nn.Linear(128 * 8 * 8, num_classes)
           self.relu = nn.ReLU()
           self.pool = nn.MaxPool2d(2)

       def forward(self, x):
           x = self.relu(self.bn1(self.conv1(x)))
           x = self.pool(x)
           x = self.relu(self.bn2(self.conv2(x)))
           x = self.pool(x)
           x = x.view(x.size(0), -1)
           return self.fc(x)

   aug = DeepAugment(X_train, y_train, X_val, y_val, model=ConvNetBN)
   best_policy = aug.optimize(iterations=50)

Model Considerations
--------------------

Training Speed
~~~~~~~~~~~~~~

Larger models take longer to train. Consider:

- Using smaller ``train_size`` and ``val_size``
- Reducing ``epochs``
- Using fewer ``iterations``
- Smaller batch sizes if memory limited

.. code-block:: python

   # Fast configuration for large models
   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       model=LargeModel,
       train_size=500,   # Small subset
       val_size=100,
       batch_size=16,    # Small batch
   )
   best_policy = aug.optimize(
       iterations=25,    # Fewer iterations
       epochs=5          # Fewer epochs
   )

Memory Usage
~~~~~~~~~~~~

If you get OOM errors:

.. code-block:: python

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       model=MyModel,
       batch_size=16,     # Reduce batch size
       train_size=500,    # Reduce data size
   )

Model Complexity
~~~~~~~~~~~~~~~~

**Rule of thumb**:

- **Small models** (< 1M params): Use as-is, fast optimization
- **Medium models** (1-10M params): Reduce train_size to 1000-2000
- **Large models** (> 10M params): Use train_size=500, fewer iterations

Pretrained Models
-----------------

Using pretrained weights:

.. code-block:: python

   import torch.nn as nn
   from torchvision.models import resnet18, ResNet18_Weights

   class PretrainedResNet(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           # Load pretrained model
           self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

           # Freeze early layers (optional)
           for param in list(self.model.parameters())[:-10]:
               param.requires_grad = False

           # Adjust for dataset
           self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

       def forward(self, x):
           return self.model(x)

   aug = DeepAugment(
       X_train, y_train, X_val, y_val,
       model=PretrainedResNet,
       learning_rate=0.001,  # Lower LR for pretrained models
   )

Multi-Task Models
-----------------

Models with multiple outputs:

.. code-block:: python

   import torch.nn as nn
   from deepaugment import DeepAugment

   class MultiTaskModel(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.features = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
           )
           self.classifier = nn.Linear(64 * 16 * 16, num_classes)

       def forward(self, x):
           features = self.features(x)
           features = features.view(features.size(0), -1)
           logits = self.classifier(features)
           return logits  # Must return logits for classification

   # DeepAugment expects single output (logits)
   aug = DeepAugment(X_train, y_train, X_val, y_val, model=MultiTaskModel)
   best_policy = aug.optimize(iterations=50)

Custom Training Logic
---------------------

For advanced use cases, you can wrap training logic:

.. code-block:: python

   import torch.nn as nn
   from deepaugment import DeepAugment

   class ModelWithCustomInit(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
           self.fc = nn.Linear(64 * 32 * 32, num_classes)

           # Custom initialization
           nn.init.kaiming_normal_(self.conv1.weight)
           nn.init.xavier_normal_(self.fc.weight)

       def forward(self, x):
           x = self.conv1(x)
           x = x.view(x.size(0), -1)
           return self.fc(x)

   aug = DeepAugment(X_train, y_train, X_val, y_val, model=ModelWithCustomInit)
   best_policy = aug.optimize(iterations=50)

Model Requirements
------------------

Your custom model must:

1. **Inherit from** ``nn.Module``
2. **Accept** ``num_classes`` in ``__init__``
3. **Return** logits (raw scores, no softmax) in ``forward()``
4. **Input shape**: ``(batch_size, 3, height, width)``
5. **Output shape**: ``(batch_size, num_classes)``

Example template:

.. code-block:: python

   import torch.nn as nn

   class MyModel(nn.Module):
       def __init__(self, num_classes=10):
           """
           Args:
               num_classes: Number of output classes
           """
           super().__init__()
           # Define layers here
           self.layers = nn.Sequential(...)

       def forward(self, x):
           """
           Args:
               x: Tensor of shape (batch_size, 3, height, width)

           Returns:
               logits: Tensor of shape (batch_size, num_classes)
           """
           return self.layers(x)

Troubleshooting
---------------

**Model not learning**
    Check learning rate, increase ``epochs``, verify data normalization

**OOM errors**
    Reduce ``batch_size``, ``train_size``, or use smaller model

**Slow training**
    Use GPU, reduce model size, or decrease ``train_size``

**Different results each run**
    Set ``random_state`` for reproducibility

**Model returns wrong shape**
    Ensure output is ``(batch_size, num_classes)`` without softmax

See Also
--------

- :doc:`cifar10` - Complete CIFAR-10 example
- :doc:`../user-guide/advanced-usage` - Advanced features
- :doc:`../api/index` - API reference
