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
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
%matplotlib osx

def _bit_round(alphabet: np.array, t: float, rad: float) -> float:
    """Rounds a quantity to the nearest atom in the (scaled) quantization alphabet.

    Parameters
    -----------
    t : float
        The value to quantize.
    rad : float
        Scaling factor for the quantization alphabet.

    Returns
    -------
    bit : float
        The quantized value.
    """

    # Scale the alphabet appropriately.
    layer_alphabet = rad * alphabet
    return layer_alphabet[np.argmin(np.abs(layer_alphabet - t))]

_, test = cifar10.load_data()
X_test, y_test = test
X_test = X_test / 255.0

num_classes = np.unique(y_test).shape[0]
y_test = to_categorical(y_test, num_classes)
input_shape = X_test[0].shape

# Load quantized model
# Best performing CNN model
model = load_model(f'/Users/elybrandadmin/Desktop/quantized_neural_networks/serialized_models/experiment_2020-06-30_09:48:23.517400/Sequential2020-07-01_00:56:47.573179')
greedy_model = load_model('/Users/elybrandadmin/Desktop/quantized_neural_networks/quantized_models/experiment_2020-07-06_13:34:07.145754/Quantized_Sequential2020-07-06_23:44:19.999083/')

_, analog_accuracy = model.evaluate(X_test, y_test, verbose=True)

# Construct MSQ Net.
MSQ_model = clone_model(model)
# Set all the weights to be equal at first. This matters for batch normalization layers.
MSQ_model.set_weights(model.get_weights())
alphabet = np.linspace(-1, 1, num=int(round(2 ** (4))))
for layer_idx, layer in enumerate(model.layers):
    if layer.__class__.__name__ in ("Dense", "Conv2D"):
        # Use the same radius as the alphabet in the corresponding layer of the Sigma Delta network.
        rad = max(
            greedy_model.layers[layer_idx].get_weights()[0].flatten()
        )
        W, b = model.layers[layer_idx].get_weights()
        Q = np.array([_bit_round(alphabet, w, rad) for w in W.flatten()]).reshape(
            W.shape
        )
        MSQ_model.layers[layer_idx].set_weights([Q, b])

MSQ_model.compile(
    optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
)
MSQ_loss, MSQ_accuracy = MSQ_model.evaluate(X_test, y_test, verbose=True)

# Compute layerwise accuracies
num_layers = len(model.layers)

# See how accuracies degrade as a function of layers quantized. Also calculate compression ratios.
MSQ_metrics = pd.DataFrame(
    {"accuracy": np.zeros(num_layers), "compression": np.zeros(num_layers)},
    index=range(num_layers),
)
greedy_metrics = pd.DataFrame(
    {"accuracy": np.zeros(num_layers), "compression": np.zeros(num_layers)},
    index=range(num_layers),
)

MSQ_tmp_net = clone_model(model)
MSQ_tmp_net.set_weights(model.get_weights())
greedy_tmp_net = clone_model(model)
greedy_tmp_net.set_weights(model.get_weights())

for layer_idx, layer in enumerate(model.layers):
    if layer.__class__.__name__ in ("Dense", "Conv2D"):
        W = layer.get_weights()[0]
        supp_W = sum(W.flatten() != 0)

        MSQ_wts = MSQ_model.layers[layer_idx].get_weights()
        MSQ_tmp_net.layers[layer_idx].set_weights(MSQ_wts)
        MSQ_tmp_net.compile(
            optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        loss, acc = MSQ_tmp_net.evaluate(X_test, y_test, verbose=False)
        MSQ_metrics["accuracy"][layer_idx] = acc
        MSQ_metrics["compression"][layer_idx] = sum(MSQ_wts[0].flatten() != 0) / supp_W

        SD_wts = greedy_model.layers[layer_idx].get_weights()
        greedy_tmp_net.layers[layer_idx].set_weights(SD_wts)
        greedy_tmp_net.compile(
            optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        loss, acc = greedy_tmp_net.evaluate(X_test, y_test, verbose=False)
        greedy_metrics["accuracy"][layer_idx] = acc
        greedy_metrics["compression"][layer_idx] = sum(SD_wts[0].flatten() != 0) / supp_W

    else:
        MSQ_metrics["accuracy"][layer_idx] = MSQ_metrics["accuracy"][layer_idx - 1]
        MSQ_metrics["compression"][layer_idx] = 1

        greedy_metrics["accuracy"][layer_idx] = greedy_metrics["accuracy"][layer_idx - 1]
        greedy_metrics["compression"][layer_idx] = 1

# Plot accuracies and compression ratios
layer_idxs = [i for i in range(num_layers) if model.layers[i].__class__.__name__ in ('Dense', 'Conv2D')]
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(layer_idxs, [greedy_metrics["accuracy"][i] for i in layer_idxs], "-o", linewidth=4, markersize=10)
ax.plot(layer_idxs, [MSQ_metrics["accuracy"][i] for i in layer_idxs], "-o", linewidth=4, markersize=10)
ax.plot(layer_idxs, [analog_accuracy for i in layer_idxs], "--", linewidth=4, markersize=10)
ax.legend(["GPFQ","MSQ", "Analog"], fontsize=12)
ax.set_title("CIFAR10 Test Accuracy vs Layers Quantized", fontsize=22)
ax.set_xticks(layer_idxs)
ax.set_yticks(np.arange(0.82, 0.9, 0.01))
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_xlabel("Layer Index", fontsize=18)
ax.set_ylabel("Test Accuracy", fontsize=18)

# Histogram the weights of the first layer
layer_idx = 2
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
W = model.layers[layer_idx].get_weights()[0]
greedy_Q = greedy_model.layers[layer_idx].get_weights()[0]
MSQ_Q = MSQ_model.layers[layer_idx].get_weights()[0]
# ax2.hist(W.flatten(), bins=32, alpha=0.3)
ax2.hist(greedy_Q.flatten(), bins=32, alpha=0.5)
ax2.hist(MSQ_Q.flatten(), bins=32, alpha=0.5)
ax2.set_xlabel("Weight Value", fontsize=18)
ax2.set_xticks(np.unique(greedy_Q.flatten())[::2])
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
ax2.set_title(f"Histogram of Layer {layer_idx} Weights", fontsize=22)
ax2.legend(["GPFQ Weights", "MSQ Weights" ], fontsize=12)
