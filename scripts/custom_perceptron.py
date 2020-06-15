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
from tensorflow.keras.initializers import (
    RandomNormal,
    GlorotUniform,
    GlorotNormal,
    RandomUniform,
)
from tensorflow.keras.models import Sequential, clone_model, save_model
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedNeuralNetwork
from sys import stdout
from os import mkdir
from itertools import chain

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

# Output shape of conv2d layer is
# np.floor((height - kernel_height + padding + stride)/stride) x ...

# Output shape of max pooling with padding 'valid' (i.e. no padding)
# np.floor((input_shape - pool_size + 1)/strides)
# Otherwise, with padding it's
# np.floor(input_shape/strides)

# Make a directory to store serialized models.
timestamp = str(pd.Timestamp.now())
serialized_model_dir = f"../serialized_models/experiment_{timestamp}".replace(" ", "_")
mkdir(serialized_model_dir)

# Here are all the parameters we iterate over. Don't go too crazy here. Training CNN's is very slow.

#TODO: Cross validate over the alphabet scalar!!

data_sets = ["mnist"]
np_seeds = [0]
tf_seeds = [0]
layer_widths = [(20,)]
rectifiers = ["relu"]
kernel_inits = [GlorotUniform]
train_batch_sizes = [128]
epochs = [1]
ignore_layers = [[]]
retrain_tries = [1]
retrain_init = ["greedy"]
bits = [np.log2(i) for i in  (3,)]
alphabet_scalars = [1]

parameter_grid = product(
    data_sets,
    np_seeds,
    tf_seeds,
    layer_widths,
    rectifiers,
    kernel_inits,
    train_batch_sizes,
    epochs,
    ignore_layers,
    retrain_tries,
    retrain_init,
    bits,
    alphabet_scalars,
)

ParamConfig = namedtuple(
    "ParamConfig",
    "data_set, np_seed, tf_seed, layer_widths, rectifier, kernel_init, "
    "train_batch_size, epochs, ignore_layers, retrain_tries, "
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
    # quant_train_size = parameters.q_train_size
    # net_train_size = train_size - quant_train_size
    # net_train_idxs = np.random.choice(train_size, size=net_train_size, replace=False)
    # quant_train_idxs = list(set(np.arange(train_size)) - set(net_train_idxs))

    # Split labels from data. Use one-hot encoding for labels.
    # MNIST ONLY: Reshape images to 28x28x1, because tensorflow is whiny.
    X_train, y_train = train
    X_test, y_test = test

    # Vectorize the images
    img_size = X_train[0].size
    X_train = X_train.reshape(X_train.shape[0], img_size)
    X_test = X_test.reshape(X_test.shape[0], img_size)

    num_classes = np.unique(y_train).shape[0]
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Futher separate training
    # X_net_train = X_train[net_train_idxs]
    # y_net_train = y_train[net_train_idxs]

    # X_quant_train = X_train[quant_train_idxs]

    try:
        # Construct a basic convolutional neural network.
        model = Sequential()
        for layer_idx, layer_width in enumerate(parameters.layer_widths):
            if layer_idx == 0:
                model.add(
                    Dense(
                        layer_width,
                        activation=parameters.rectifier,
                        kernel_initializer=parameters.kernel_init(),
                        use_bias=True,
                        input_dim=img_size,
                    )
                )
            else:
                model.add(
                    Dense(
                        layer_width,
                        activation=parameters.rectifier,
                        kernel_initializer=parameters.kernel_init(),
                        use_bias=True,
                    )
                )
            model.add(BatchNormalization())

        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
    except ValueError:
        # Inconsistent paramter configuration.
        logger.warning(
            "Inconsistent parameter configuration. Skipping to next parameter configuration."
        )
        return

    for train_idx in range(parameters.retrain_tries):

        if train_idx == 0:
            # Initialize the weights using the kernel_initializer provided.
            logger.info(
                f"Training with parameters {parameters}. Training iteration {train_idx+1} of {parameters.retrain_tries}."
            )
            history = model.fit(
                X_train,
                y_train,
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

            if parameters.retrain_init == "greedy":
                model.set_weights(my_quant_net.quantized_net.get_weights())
            if parameters.retrain_init == "msq":
                # TODO: Make a MSQ class to handle MSQ networks please.
                pass
            history = model.fit(
                X_train,
                y_train,
                batch_size=parameters.train_batch_size,
                epochs=parameters.epochs,
                verbose=True,
                validation_split=0.20,
            )
        get_data = (sample for sample in X_train)
        for i in range(len(parameters.layer_widths)+1):
            # Chain together iterators over the entire training set. This is so each layer uses
            # the entire training data.
            get_data = chain(get_data, (sample for sample in X_train))
        batch_size = X_train.shape[0]
        my_quant_net = QuantizedNeuralNetwork(
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

    model_timestamp = str(pd.Timestamp.now()).replace(" ", "_")
    model_name = model.__class__.__name__ + str(pd.Timestamp.now()).replace(" ", "_")
    save_model(model, f"{serialized_model_dir}/{model_name}")

    trial_metrics = pd.DataFrame(
        {
            "data_set": parameters.data_set,
            "np_seed": parameters.np_seed,
            "tf_seed": parameters.tf_seed,
            "serialized_model": model_name,
            "layer_widths": [parameters.layer_widths],
            "rectifier": parameters.rectifier,
            "train_batch_size": parameters.train_batch_size,
            "epochs": parameters.epochs,
            "q_train_size": batch_size,
            "ignore_layers": [parameters.ignore_layers],
            "retrain_tries": parameters.retrain_tries,
            "retrain_init": parameters.retrain_init,
            "bits": parameters.bits,
            "alphabet_scalar": parameters.alphabet_scalar,
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
