import numpy as np
import pandas as pd
import logging
from itertools import product
from collections import namedtuple
from tensorflow.random import set_seed
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Flatten,
)
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Sequential, clone_model, save_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedNeuralNetwork
from sys import stdout
from os import mkdir
from itertools import chain

NP_SEED = 0
TF_SEED = 0
EPOCHS = 100
LAYER_WIDTHS = (500, 300)
VALIDATION_SPLIT = 0.2

# Write logs to file and to stdout. Overwrite previous log file.
fh = logging.FileHandler("../train_logs/model_training.log", mode="w+")
fh.setLevel(logging.INFO)
sh = logging.StreamHandler(stream=stdout)
sh.setLevel(logging.INFO)

# Only use the logger in this module.
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

# Make a directory to store serialized models.
timestamp = str(pd.Timestamp.now())
serialized_model_dir = f"../serialized_models/"

# Set the random seeds for numpy and tensorflow.
set_seed(0)
np.random.seed(0)

# Split training from testing
train, test = mnist.load_data()
train_size = train[0].shape[0]

# Split labels from data. Use one-hot encoding for labels.
X_train, y_train = train
X_test, y_test = test

# Use one-hot encoding for the labels.
num_classes = np.unique(y_train).shape[0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the model.
model = Sequential()
model.add(Flatten(input_shape=X_train[0].shape,))
for layer_idx, layer_width in enumerate(LAYER_WIDTHS):
    model.add(
        Dense(
            layer_width,
            activation="relu",
            kernel_initializer=GlorotUniform(),
            use_bias=True,
        )
    )
    model.add(BatchNormalization())
model.add(Dense(num_classes, activation="softmax"))
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=EPOCHS,
    verbose=True,
    validation_split=VALIDATION_SPLIT,
)

analog_loss, analog_accuracy = model.evaluate(X_test, y_test, verbose=True)
logger.info(f"Analog model test accuracy = {analog_accuracy:.2f}")

model_timestamp = str(pd.Timestamp.now()).replace(" ", "_").replace(":","").replace(".","")
model_name = model.__class__.__name__ + model_timestamp
save_model(model, f"{serialized_model_dir}/MNIST_{model_name}")
