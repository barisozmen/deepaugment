# (C) 2017 Tito Ingargiola
# (C) 2024 Peter Norvig

from tensorflow.keras import layers, models

def WideResidualNetwork(depth=28, width=8, dropout_rate=0.0, input_shape=None, classes=10):
    if (depth - 4) % 6 != 0:
        raise ValueError("Depth must be 6n + 4.")

    n_blocks = (depth - 4) // 6
    img_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(img_input)

    for _ in range(n_blocks):
        x = _conv_block(x, 16 * width, dropout_rate)
    x = layers.MaxPooling2D((2, 2))(x)

    for _ in range(n_blocks):
        x = _conv_block(x, 32 * width, dropout_rate)
    x = layers.MaxPooling2D((2, 2))(x)

    for _ in range(n_blocks):
        x = _conv_block(x, 64 * width, dropout_rate)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, activation="softmax")(x)

    return models.Model(img_input, x, name="wide-resnet")

def _conv_block(x, n_filters, dropout_rate):
    shortcut = x
    if x.shape[-1] != n_filters:
        shortcut = layers.Conv2D(n_filters, (1, 1), padding="same")(x)

    x = layers.Conv2D(n_filters, (3, 3), padding="same", activation="relu")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(n_filters, (3, 3), padding="same", activation="relu")(x)

    return layers.add([shortcut, x])
