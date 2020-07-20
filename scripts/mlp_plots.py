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

_, test = mnist.load_data()
X_test, y_test = test

num_classes = np.unique(y_test).shape[0]
y_test = to_categorical(y_test, num_classes)
input_shape = X_test[0].shape

# Load analog and quantized model.
model = load_model(f'/Users/elybrandadmin/Desktop/quantized_neural_networks/serialized_models/MNIST_Sequential2020-07-14_112220282408')
analog_loss, analog_accuracy = model.evaluate(X_test, y_test, verbose=True)

# Load the model metrics
df = pd.read_csv("../model_metrics/mnist_model_metrics_2020-07-15_121741452415.csv")

# Make a plot of the test accuracies vs alphabet scalars.
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.plot(df['alphabet_scalar'], df['sd_test_acc'], '-o', linewidth=4, markersize=10)
ax.plot(df['alphabet_scalar'], df['msq_test_acc'], '-o', linewidth=4, markersize=10)
ax.plot(df['alphabet_scalar'], [analog_accuracy for c in df['alphabet_scalar']], '--', linewidth=4)
ax.legend(['Greedy', 'MSQ', 'Analog'], fontsize=12)
ax.set_xlabel("Alphabet Scalar", fontsize=18)
ax.set_ylabel("Test Accuracy", fontsize=18)
ax.set_title("MNIST Test Accuracy vs. Alphabet Scalar", fontsize=22)
ax.set_xticks(df['alphabet_scalar'])
ax.set_yticks(np.arange(0,1.1,0.1))
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# Load the best greedy quantized network
greedy_model = load_model(f'/Users/elybrandadmin/Desktop/quantized_neural_networks/quantized_models/quantized_mnist_scaler3_2020-07-15_125211560640')
q_loss, q_accuracy = greedy_model.evaluate(X_test, y_test, verbose=True)

# Grab the best alphabet scalar for MSQ
MSQ_scalar = df.loc[df['msq_test_acc'] == max(df['msq_test_acc'])]['alphabet_scalar'].values[0]

# Construct MSQ Net.
MSQ_model = clone_model(model)
# Set all the weights to be equal at first. This matters for batch normalization layers.
MSQ_model.set_weights(model.get_weights())
alphabet = np.array([-1, 0, 1])
for layer_idx, layer in enumerate(model.layers):
    if layer.__class__.__name__ in ("Dense", "Conv2D"):
        W, b = model.layers[layer_idx].get_weights()
        rad = MSQ_scalar * np.median(abs(W.flatten()))
        Q = np.array([_bit_round(alphabet, w, rad) for w in W.flatten()]).reshape(
            W.shape
        )
        MSQ_model.layers[layer_idx].set_weights([Q, b])

MSQ_model.compile(
    optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
)
MSQ_loss, MSQ_accuracy = MSQ_model.evaluate(X_test, y_test, verbose=True)

# See how accuracies degrade as a function of layers quantized.
num_layers = len(model.layers)
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
    if layer.__class__.__name__ in ("Dense", ):
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

# Remove rows of dataframe that don't correspond to dense layers.
greedy_metrics = greedy_metrics.query("accuracy > 0")
MSQ_metrics = MSQ_metrics.query("accuracy > 0")

# Plot accuracies and compression ratios
layer_idxs = [i for i in range(num_layers) if model.layers[i].__class__.__name__ in ('Dense', 'Conv2D')]
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(greedy_metrics.index, greedy_metrics['accuracy'], "-o", linewidth=4, markersize=10)
ax.plot(MSQ_metrics.index, MSQ_metrics['accuracy'], "-o", linewidth=4, markersize=10)
ax.plot(greedy_metrics.index, [analog_accuracy for c in greedy_metrics.index], '--', linewidth=4)
ax.legend(["Greedy", "MSQ", 'Analog'], fontsize=12)
ax.set_title("MNIST Test Accuracy vs. Layers Quantized", fontsize=22)
ax.set_xticks(layer_idxs)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
# ax.set_ylim([0.8, 0.95])
ax.set_xlabel("Layer Index", fontsize=18)
ax.set_ylabel("Test Accuracy", fontsize=18)

# Histogram the weights of the last layer
layer_idx = 5
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
W = model.layers[layer_idx].get_weights()[0]
greedy_Q = greedy_model.layers[layer_idx].get_weights()[0]
MSQ_Q = MSQ_model.layers[layer_idx].get_weights()[0]
# ax2.hist(W.flatten(), bins=32, alpha=0.3)
ax2.hist(MSQ_Q.flatten(), bins=32, alpha=0.5)
ax2.hist(greedy_Q.flatten(), bins=32, alpha=0.5)
ax2.set_xlabel("Weight Value", fontsize=18)
ax2.set_xticks(np.unique(greedy_Q.flatten()))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.set_title(f"Histogram of Layer {layer_idx} Weights", fontsize=22)
ax2.legend([ "MSQ Weights", "Greedy Weights"], fontsize=12)
