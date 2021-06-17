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
from tensorflow.keras.models import Sequential, clone_model, save_model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedNeuralNetwork, MNISTSequence, _bit_round_parallel
from sys import stdout, argv
from os import mkdir
from itertools import chain
from time import time

# Write logs to file and to stdout. Overwrite previous log file.
fh = logging.FileHandler("../train_logs/model_quantizing.log", mode="w+")
fh.setLevel(logging.INFO)
sh = logging.StreamHandler(stream=stdout)
sh.setLevel(logging.INFO)

# Only use the logger in this module.
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

# Set the parameters for model quantization.
quant_train_size = 25000
data_sets = ["mnist"]
bits = [np.log2(i) for i in (3,)]
alphabet_scalars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Make an iterable parameter grid.
parameter_grid = product(
    data_sets,
    bits,
    alphabet_scalars,
)
ParamConfig = namedtuple("ParamConfig", "data_set, bits, alphabet_scalar")
param_iterable = (ParamConfig(*config) for config in parameter_grid)

# Load analog model from command line argument
analog_model = argv[1]
model = load_model(f"../serialized_models/{analog_model}")

# Split training from testing
train, test = mnist.load_data()
train_size = train[0].shape[0]

# Split labels from data. 
X_train, y_train = train
X_test, y_test = test

# Use one-hot encoding for labels.
num_classes = np.unique(y_train).shape[0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def train_network(parameters: ParamConfig) -> pd.DataFrame:

    _, analog_accuracy = model.evaluate(X_test, y_test, verbose=True)

    # Determine how many layers you need to quantize.
    num_layers = sum([layer.__class__.__name__ in ('Dense',) for layer in model.layers])

    get_data = MNISTSequence(X_train[0:quant_train_size], y_train[0:quant_train_size], batch_size=quant_train_size)
    batch_size = quant_train_size
    my_quant_net = QuantizedNeuralNetwork(
        network=model,
        batch_size=batch_size,
        get_data=get_data,
        logger=logger,
        bits=parameters.bits,
        alphabet_scalar=parameters.alphabet_scalar,
    )
    tic = time()
    my_quant_net.quantize_network()
    quantization_time = time() - tic

    my_quant_net.quantized_net.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    _, q_accuracy = my_quant_net.quantized_net.evaluate(X_test, y_test, verbose=True)

    # Serialize the greedy network.
    model_timestamp = str(pd.Timestamp.now()).replace(" ", "_").replace(":","").replace(".","")
    model_name = f"quantized_mnist_scaler{parameters.alphabet_scalar}_{model_timestamp}"
    save_model(my_quant_net.quantized_net, f"../quantized_models/{model_name}")

    # Construct MSQ Net.
    MSQ_model = clone_model(model)
    # Set all the weights to be equal at first. This matters for batch normalization layers.
    MSQ_model.set_weights(model.get_weights())
    for layer_idx, layer in enumerate(model.layers):
        if (
            layer.__class__.__name__ in ("Dense", "Conv2D")
        ):
            # Use the same radius as the alphabet in the corresponding layer of the Sigma Delta network.
            W, b = model.layers[layer_idx].get_weights()
            rad = parameters.alphabet_scalar * np.median(np.abs(W.flatten()))
            layer_alphabet = rad*my_quant_net.alphabet
            Q = np.array([_bit_round_parallel(w, layer_alphabet) for w in W.flatten()]).reshape(
                W.shape
            )
            MSQ_model.layers[layer_idx].set_weights([Q, b])

    MSQ_model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    _, MSQ_accuracy = MSQ_model.evaluate(X_test, y_test, verbose=True)

    trial_metrics = pd.DataFrame(
        {
            "data_set": parameters.data_set,
            "analog_model": analog_model,
            "serialized_quantized_model": model_name,
            "q_train_size": batch_size,
            "bits": parameters.bits,
            "alphabet_scalar": parameters.alphabet_scalar,
            "analog_test_acc": analog_accuracy,
            "sd_test_acc": q_accuracy,
            "msq_test_acc": MSQ_accuracy,
            "quantization_time": quantization_time
        },
        index=[model_timestamp],
    )

    return trial_metrics


if __name__ == "__main__":

    # Store results in csv file.
    timestamp = str(pd.Timestamp.now()).replace(" ", "_").replace(":","").replace(".","")
    file_name = data_sets[0] + "_model_metrics_" + timestamp
    # Timestamp adds a space. Replace it with _
    file_name = file_name.replace(" ", "_")

    for idx, params in enumerate(param_iterable):
        trial_metrics = train_network(params)
        if idx == 0:
            # add the header
            trial_metrics.to_csv(f"../model_metrics/{file_name}.csv", mode="a")
        else:
            trial_metrics.to_csv(
                f"../model_metrics/{file_name}.csv", mode="a", header=False
            )
