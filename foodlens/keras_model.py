"""Keras twin of FoodCNN (same layers, same input scaling, logits output)."""
import keras
from keras import layers

from .classes import NUM_CLASSES


def build_keras_cnn(size: int = 128, num_classes: int = NUM_CLASSES) -> keras.Model:
    x = inp = keras.Input((size, size, 3))
    for ch in (32, 64, 128):
        x = layers.Conv2D(ch, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    return keras.Model(inp, layers.Dense(num_classes)(x))  # raw logits
