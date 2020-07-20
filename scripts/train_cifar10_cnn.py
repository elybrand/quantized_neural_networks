import numpy as np
import pandas as pd
import logging
from tensorflow.random import set_seed
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, clone_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedCNN
from itertools import chain
from sys import stdout
from os import mkdir

NP_SEED = 0
TF_SEED = 0
EPOCHS = 400

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

serialized_model_dir = f"../serialized_models/"

# Set the random seeds for numpy and tensorflow.
set_seed(0)
np.random.seed(0)

train, test = cifar10.load_data()
X_train, y_train = train
X_test, y_test = test

# Normalize pixel values.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# Use one-hot encoding for labels.
num_classes = np.unique(y_train).shape[0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the convolutional neural network.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Augment training data with shifts and flips.
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train = datagen.flow(X_train, y_train, batch_size=64)

# Train the model. Note that fit() does not train on the validation data.
history = model.fit(
    it_train,
    epochs=EPOCHS,
    verbose=True,
    validation_data=(X_test, y_test),
    steps_per_epoch=int(X_train.shape[0] / 64)
)

# Save the model to disk.
model_timestamp = str(pd.Timestamp.now()).replace(" ", "_").replace(":","").replace(".","")
model_name = model.__class__.__name__ + model_timestamp
save_model(model, f"{serialized_model_dir}/CIFAR10_{model_name}")

loss, accuracy = model.evaluate(X_test, y_test, verbose=True)

trial_metrics = pd.DataFrame(
    {
        "data_set": "cifar10",
        "serialized_model": model_name,
        "np_seed": NP_SEED,
        "tf_seed": TF_SEED,
        "epochs": EPOCHS,
        "analog_test_acc": accuracy,
    },
    index=[model_timestamp],
)
