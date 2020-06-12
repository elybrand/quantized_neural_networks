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
)
from tensorflow.keras.initializers import (
    RandomNormal,
    GlorotUniform,
    GlorotNormal,
    RandomUniform,
)
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedCNN
from sys import stdout

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

# Set the random seeds for numpy and tensorflow.
np_seed = 0
tf_seed = 0
set_seed(tf_seed)
np.random.seed(np_seed)

# Here are all the parameters we iterate over. Don't go too crazy here. Training CNN's is very slow.
data_sets = ["mnist"]
trial_idxs = [0, 1, 2, 3, 4]
rectifiers = ["relu"]
kernel_inits = [GlorotUniform]
kernel_sizes = [3,  5,  7]
strides = [2]
train_batch_sizes = [128]
epochs = [10]
q_train_sizes = [10**4]
ignore_layers = [[]]
retrain_tries = [1, 2, 3]
bits = [np.log2(i) for i in range(3, 8)]

parameter_grid = product(
    data_sets,
    trial_idxs,
    [np_seed],
    [tf_seed],
    rectifiers,
    kernel_inits,
    kernel_sizes,
    strides,
    train_batch_sizes,
    epochs,
    q_train_sizes,
    ignore_layers,
    retrain_tries,
    bits,
)

ParamConfig = namedtuple(
    "ParamConfig",
    "data_set, trial_idx, np_seed, tf_seed, rectifier, kernel_init, kernel_size, stride, train_batch_size, epochs, q_train_size, ignore_layer, retrain_tries, bits",
)
param_iterable = (ParamConfig(*config) for config in parameter_grid)

# Build a data frame to keep track of each trial's metrics.
model_metrics = pd.DataFrame(
    {
        "data_set": [],
        "np_seed": [],
        "tf_seed": [],
        "serialized_model": [],
        "trial_idx": [],
        "rectifier": [],
        "kernel_init": [],
        "kernel_size": [],
        "strides": [],
        "train_batch_size": [],
        "epochs": [],
        "q_train_size": [],
        "retrain_tries": [],
        "bits": [],
        "analog_test_acc": [],
        "sd_test_acc": [],
        "msq_test_acc": [],
    },
    index=[],
)


def train_network(parameters: ParamConfig) -> pd.DataFrame:

    # Split training from testing
    train, test = globals()[parameters.data_set].load_data()

    # Split the training data into two populations. One for training the network and
    # one for training the quantization. For now, split it evenly.
    train_size = train[0].shape[0]
    quant_train_size = parameters.q_train_size
    net_train_size = train_size - quant_train_size
    net_train_idxs = np.random.choice(train_size, size=net_train_size, replace=False)
    quant_train_idxs = list(set(np.arange(train_size)) - set(net_train_idxs))

    # Split labels from data. Use one-hot encoding for labels.
    # MNIST ONLY: Reshape images to 28x28x1, because tensorflow is whiny.
    X_train, y_train = train
    X_test, y_test = test

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

    # Futher separate training
    X_net_train = X_train[net_train_idxs]
    y_net_train = y_train[net_train_idxs]

    X_quant_train = X_train[quant_train_idxs]

    # Construct a basic convolutional neural network.
    model = Sequential()
    model.add(
        Conv2D(
            filters=64,
            kernel_size=parameters.kernel_size,
            strides=(parameters.stride, parameters.stride),
            activation=parameters.rectifier,
            kernel_initializer=parameters.kernel_init(),
            input_shape=input_shape,
        )
    )

    #TODO: Max pooling, 2x2 pixels with stride of 2.

    model.add(BatchNormalization())
    model.add(
        Conv2D(
            filters=32,
            kernel_size=parameters.kernel_size,
            strides=(parameters.stride, parameters.stride),
            kernel_initializer=parameters.kernel_init(),
            activation=parameters.rectifier,
        )
    )

    #TODO: Max pooling, 2x2 pixels with stride of 2

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    # Not too many epochs...training CNN's appears to be quite slow.
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    for train_idx in range(parameters.retrain_tries):

        if train_idx == 0:
            # Initialize the weights using the kernel_initializer provided.
            logger.info(
                f"Training with parameters {parameters}. Training iteration {train_idx+1} of {parameters.retrain_tries}."
            )
            history = model.fit(
                X_net_train,
                y_net_train,
                batch_size=parameters.train_batch_size,
                epochs=parameters.epochs,
                verbose=True,
                validation_split=0.20,
            )
        else:
            # Initialize using the quantized network's weights.
            logger.info(
                f"Retraining with parameters {parameters}. Training iteration {train_idx+1} of {parameters.retrain_tries}."
            )
            model.set_weights(my_quant_net.quantized_net.get_weights())
            history = model.fit(
                X_net_train,
                y_net_train,
                batch_size=parameters.train_batch_size,
                epochs=parameters.epochs,
                verbose=True,
                validation_split=0.20,
            )

        # Quantize the network.
        get_data = (sample for sample in X_quant_train)
        # Make it so all data are used.
        batch_size = int(np.floor(quant_train_size / (3)))
        is_debug = False
        # TODO: add ignore layer sometime in the future.
        my_quant_net = QuantizedCNN(
            network=model,
            batch_size=batch_size,
            get_data=get_data,
            logger=logger,
            is_debug=is_debug,
            bits=parameters.bits,
        )
        my_quant_net.quantize_network()

    my_quant_net.quantized_net.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    analog_loss, analog_accuracy = model.evaluate(X_test, y_test, verbose=True)
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

    timestamp = str(pd.Timestamp.now())
    model_name = model_name = model.__class__.__name__ + timestamp

    # TODO: serialize model

    trial_metrics = pd.DataFrame(
        {
            "data_set": parameters.data_set,
            "np_seed": parameters.np_seed,
            "tf_seed": parameters.tf_seed,
            "serialized_model": model_name,
            "trial_idx": parameters.trial_idx,
            "rectifier": parameters.rectifier,
            "kernel_init": parameters.kernel_init.__name__,
            "kernel_size": parameters.kernel_size,
            "strides": parameters.stride,
            "train_batch_size": parameters.train_batch_size,
            "epochs": parameters.epochs,
            "q_train_size": parameters.q_train_size,
            "retrain_tries": parameters.retrain_tries,
            "bits": parameters.bits,
            "analog_test_acc": analog_accuracy,
            "sd_test_acc": q_accuracy,
            "msq_test_acc": MSQ_accuracy,
        },
        index=[timestamp],
    )

    return trial_metrics


if __name__ == "__main__":

    # Store results in csv file.
    file_name = data_sets[0] + "_model_metrics_" + str(pd.Timestamp.now())
    for idx, params in enumerate(param_iterable):
        trial_metrics = train_network(params)
        if idx == 0:
            # add the header
            trial_metrics.to_csv(f"../model_metrics/{file_name}.csv", mode="a")
        else:
            trial_metrics.to_csv(
                f"../model_metrics/{file_name}.csv", mode="a", header=False
            )
