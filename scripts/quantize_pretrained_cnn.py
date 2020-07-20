import numpy as np
import pandas as pd
import logging
from itertools import product
from collections import namedtuple
from tensorflow.random import set_seed
from tensorflow.keras.models import load_model, clone_model, save_model
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedCNN
from itertools import chain
from sys import stdout
from os import mkdir

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

# Make a directory to store serialized models.
timestamp = str(pd.Timestamp.now())
serialized_model_dir = f"../quantized_models/experiment_{timestamp}".replace(" ", "_")
mkdir(serialized_model_dir)

pretrained_model = ['experiment_2020-06-30_09:48:23.517400/Sequential2020-07-01_00:56:47.573179']
data_sets = ["cifar10"]
q_train_sizes = [5000]
ignore_layers = [[]]
bits = [np.log2(i) for i in  (3, 4, 8, 16)]
alphabet_scalars = [2, 3, 4]

parameter_grid = product(
    pretrained_model,
    data_sets,
    q_train_sizes,
    ignore_layers,
    bits,
    alphabet_scalars,
)

ParamConfig = namedtuple(
    "ParamConfig",
    "pretrained_model, data_set, q_train_size, ignore_layers, bits, alphabet_scalar",
)
param_iterable = (ParamConfig(*config) for config in parameter_grid)

def quantize_network(parameters: ParamConfig) -> pd.DataFrame:

    # Split training from testing
    train, test = globals()[parameters.data_set].load_data()

    train_size = train[0].shape[0]
    quant_train_size = parameters.q_train_size

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

    # Load the model.
    model = load_model(f'/Users/elybrandadmin/Desktop/quantized_neural_networks/serialized_models/{parameters.pretrained_model}')
    analog_loss, analog_accuracy = model.evaluate(X_test, y_test, verbose=True)
    logger.info(f"Analog network test accuracy = {analog_accuracy:.2f}")

    # Find out how many layers you're going to quantize.
    layer_names = np.array([layer.__class__.__name__ for layer in model.layers])
    num_layers_to_quantize = sum((layer_names == 'Dense') + (layer_names == 'Conv2D'))

    get_data = (sample for sample in X_train[0:quant_train_size])
    for i in range(num_layers_to_quantize):
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
    # Save the quantized model.
    model_timestamp = str(pd.Timestamp.now()).replace(" ", "_")
    model_name = model.__class__.__name__ + str(pd.Timestamp.now()).replace(" ", "_")
    save_model(my_quant_net.quantized_net, f"{serialized_model_dir}/Quantized_{model_name}")

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
            Q = np.array([my_quant_net._bit_round(w, rad) for w in W.flatten()]).reshape(
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
            "serialized_model": parameters.pretrained_model,
            "q_train_size": parameters.q_train_size,
            "ignore_layers": [parameters.ignore_layers],
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
        trial_metrics = quantize_network(params)

        if idx == 0:
            # add the header
            trial_metrics.to_csv(f"../model_metrics/{file_name}.csv", mode="a")
        else:
            trial_metrics.to_csv(
                f"../model_metrics/{file_name}.csv", mode="a", header=False
            )