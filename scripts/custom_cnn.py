import numpy as np
import pandas as pd
import logging
from itertools import product
from collections import namedtuple
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
from tensorflow.keras.initializers import (
    RandomNormal,
    GlorotUniform,
    GlorotNormal,
    RandomUniform,
)
from tensorflow.keras.models import Sequential, clone_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedCNN
from itertools import chain
from sys import stdout
from os import mkdir

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
serialized_model_dir = f"../serialized_models/experiment_{timestamp}".replace(" ", "_")
mkdir(serialized_model_dir)

# Here are all the parameters we iterate over.

data_sets = ["cifar10"]
np_seeds = [0]
tf_seeds = [0]
kernels_per_layer = [(64, 32)]
rectifiers = ["relu"]
kernel_inits = [GlorotUniform]
conv_kernel_sizes = [
    (7, 3),
]  # All kernels assumed to be square. Tuples indicate shapes per layer.
conv_strides = [
    (2, 2)
]  # Strides assumed to be equal along all dimensions. Tuples indicate shapes per layer.
dropout_rates = [(0.2, 0.2)]#, (0.35, 0.35), (0.5, 0.5)]
pool_sizes = [(2,2)]
pool_strides = [(2, 2)]
train_batch_sizes = [128]
epochs = [50]
q_train_sizes = [10]
ignore_layers = [[]]
retrain_tries = [1]
retrain_init = ["greedy"]
bits = [np.log2(i) for i in  (3,)]
alphabet_scalars = [2]

parameter_grid = product(
    data_sets,
    np_seeds,
    tf_seeds,
    kernels_per_layer,
    rectifiers,
    kernel_inits,
    conv_kernel_sizes,
    conv_strides,
    dropout_rates,
    pool_sizes,
    pool_strides,
    train_batch_sizes,
    epochs,
    q_train_sizes,
    ignore_layers,
    retrain_tries,
    retrain_init,
    bits,
    alphabet_scalars,
)

ParamConfig = namedtuple(
    "ParamConfig",
    "data_set, np_seed, tf_seed, kernels_per_layer, rectifier, kernel_init, conv_kernel_sizes, conv_strides, "
    "dropout_rates, pool_sizes, pool_strides, train_batch_size, epochs, q_train_size, ignore_layers, retrain_tries,"
    "retrain_init, bits, alphabet_scalar",
)
param_iterable = (ParamConfig(*config) for config in parameter_grid)

# Build a data frame to keep track of each trial's metrics.
model_metrics = pd.DataFrame(
    {
        "data_set": [],
        "np_seed": [],
        "tf_seed": [],
        "serialized_model": [],
        "kernels_per_layer": [],
        "rectifier": [],
        "kernel_init": [],
        "conv_kernel_sizes": [],
        "conv_strides": [],
        "dropout_rates": [],
        "pool_sizes": [],
        "pool_strides": [],
        "train_batch_size": [],
        "epochs": [],
        "q_train_size": [],
        "ignore_layers": [],
        "retrain_tries": [],
        "retrain_init": [],
        "bits": [],
        "alphabet_scalar": [],
        "analog_test_acc": [],
        "sd_test_acc": [],
        "msq_test_acc": [],
    },
    index=[],
)


def train_network(parameters: ParamConfig) -> pd.DataFrame:

    # Set the random seeds for numpy and tensorflow.
    set_seed(parameters.tf_seed)
    np.random.seed(parameters.np_seed)

    # Split training from testing
    train, test = globals()[parameters.data_set].load_data()

    # Split the training data into two populations. One for training the network and
    # one for training the quantization. For now, split it evenly.
    train_size = train[0].shape[0]
    quant_train_size = parameters.q_train_size
    # net_train_size = train_size - quant_train_size
    # net_train_idxs = np.random.choice(train_size, size=net_train_size, replace=False)
    # quant_train_idxs = list(set(np.arange(train_size)) - set(net_train_idxs))

    X_train, y_train = train
    X_test, y_test = test

    # Normalize pixel values.
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # MNIST only
    if parameters.data_set == "mnist":
        train_shape = X_train.shape
        test_shape = X_test.shape
        X_train = X_train.reshape(train_shape[0], train_shape[1], train_shape[2], 1)
        X_test = X_test.reshape(test_shape[0], test_shape[1], test_shape[2], 1)

    num_classes = np.unique(y_train).shape[0]
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    input_shape = X_train[0].shape

    try:
        # # Construct a basic convolutional neural network.
        # model = Sequential()
        # # model.add(Dropout(0.2, input_shape=input_shape))
        # model.add(
        #     Conv2D(
        #         filters=parameters.kernels_per_layer[0],
        #         kernel_size=parameters.conv_kernel_sizes[0],
        #         strides=(parameters.conv_strides[0], parameters.conv_strides[0]),
        #         activation=parameters.rectifier,
        #         kernel_initializer=parameters.kernel_init(),
        #         input_shape=input_shape,
        #     )
        # )

        # model.add(
        #     MaxPooling2D(
        #         pool_size=(parameters.pool_sizes[0], parameters.pool_sizes[0]),
        #         strides=(parameters.pool_strides[0], parameters.pool_strides[0]),
        #         padding="valid",
        #     )
        # )

        # model.add(BatchNormalization())
        # model.add(Dropout(parameters.dropout_rates[0]))
        # model.add(
        #     Conv2D(
        #         filters=parameters.kernels_per_layer[1],
        #         kernel_size=parameters.conv_kernel_sizes[1],
        #         strides=(parameters.conv_strides[1], parameters.conv_strides[1]),
        #         kernel_initializer=parameters.kernel_init(),
        #         activation=parameters.rectifier,
        #     )
        # )

        # model.add(
        #     MaxPooling2D(
        #         pool_size=(parameters.pool_sizes[1], parameters.pool_sizes[1]),
        #         strides=(parameters.pool_strides[1], parameters.pool_strides[1]),
        #         padding="valid",
        #     )
        # )

        # model.add(BatchNormalization())
        # model.add(Flatten())
        # model.add(Dropout(parameters.dropout_rates[1]))
        # model.add(Dense(10, activation="softmax"))

        # model.compile(
        #     optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        # )

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Dropout(0.2))
        # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Dropout(0.3))
        # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Dropout(0.4))
        model.add(Flatten())
        # model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        it_train = datagen.flow(X_train, y_train, batch_size=64)

    except ValueError:
        # Inconsistent paramter configuration.
        logger.warning(
            "Inconsistent parameter configuration. Skipping to next parameter configuration."
        )
        return

    # Initialize the weights using the kernel_initializer provided.
    logger.info(
        f"Training with parameters {parameters}. Training iteration {train_idx+1} of {parameters.retrain_tries}."
    )
    history = model.fit(
        it_train,
        epochs=1,
        verbose=True,
        validation_data=(X_test, y_test),
        steps_per_epoch=int(X_train.shape[0] / 64)
    )


    model_timestamp = str(pd.Timestamp.now()).replace(" ", "_")
    model_name = model.__class__.__name__ + str(pd.Timestamp.now()).replace(" ", "_")
    save_model(model, f"{serialized_model_dir}/{model_name}")

    analog_loss, analog_accuracy = model.evaluate(X_test, y_test, verbose=True)
    logger.info(f"Analog network test accuracy = {analog_accuracy:.2f}")

    get_data = (sample for sample in X_train[0:quant_train_size])
    for i in range(len(parameters.kernels_per_layer) + 1):
        # Chain together iterators over the entire training set. This is so each layer uses
        # the entire training data.
        get_data = chain(get_data, (sample for sample in X_train[0:quant_train_size]))
    batch_size = quant_train_size
    my_quant_net = QuantizedCNN(
        network=model,
        batch_size=batch_size,
        get_data=get_data,
        logger=logger,
        bits=parameters.bits,
        alphabet_scalar=parameters.alphabet_scalar,
    )
    my_quant_net.quantize_network()

    my_quant_net.quantized_net.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    q_loss, q_accuracy = my_quant_net.quantized_net.evaluate(X_test, y_test, verbose=True)

    # Construct MSQ Net.
    MSQ_model = clone_model(model)
    # Set all the weights to be equal at first. This matters for batch normalization layers.
    MSQ_model.set_weights(model.get_weights())
    for layer_idx, layer in enumerate(model.layers):
        if layer.__class__.__name__ in ("Dense", "Conv2D"):
            # Use the same radius as the alphabet in the corresponding layer of the Sigma Delta network.
            rad = max(
                my_quant_net.quantized_net.layers[layer_idx].get_weights()[0].flatten()
            )
            W, b = model.layers[layer_idx].get_weights()
            Q = np.array([my_quant_net.bit_round(w, rad) for w in W.flatten()]).reshape(
                W.shape
            )
            MSQ_model.layers[layer_idx].set_weights([Q, b])

    MSQ_model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    MSQ_loss, MSQ_accuracy = MSQ_model.evaluate(X_test, y_test, verbose=True)

    trial_metrics = pd.DataFrame(
        {
            "data_set": parameters.data_set,
            "np_seed": parameters.np_seed,
            "tf_seed": parameters.tf_seed,
            "serialized_model": model_name,
            "kernels_per_layer": [parameters.kernels_per_layer],
            "rectifier": parameters.rectifier,
            "kernel_init": parameters.kernel_init.__name__,
            "conv_kernel_sizes": [parameters.conv_kernel_sizes],
            "conv_strides": [parameters.conv_strides],
            "dropout_rates": [parameters.dropout_rates],
            "pool_sizes": [parameters.pool_sizes],
            "pool_strides": [parameters.pool_strides],
            "train_batch_size": parameters.train_batch_size,
            "epochs": parameters.epochs,
            "q_train_size": parameters.q_train_size,
            "ignore_layers": [parameters.ignore_layers],
            "retrain_tries": parameters.retrain_tries,
            "retrain_init": parameters.retrain_init,
            "bits": parameters.bits,
            "analog_test_acc": analog_accuracy,
            "sd_test_acc": q_accuracy,
            "msq_test_acc": MSQ_accuracy,
        },
        index=[model_timestamp],
    )

    return trial_metrics


if __name__ == "__main__":

    # Store results in csv file.
    file_name = data_sets[0] + "_model_metrics_" + timestamp
    # Timestamp adds a space. Replace it with _
    file_name = file_name.replace(" ", "_")
    for idx, params in enumerate(param_iterable):
        trial_metrics = train_network(params)

        if trial_metrics is None:
            # Skip this configuration. It's inconsistent.
            continue

        if idx == 0:
            # add the header
            trial_metrics.to_csv(f"../model_metrics/{file_name}.csv", mode="a")
        else:
            trial_metrics.to_csv(
                f"../model_metrics/{file_name}.csv", mode="a", header=False
            )
