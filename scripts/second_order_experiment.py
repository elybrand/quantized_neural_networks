import numpy as np
import pandas as pd
from math import sqrt, log
import scipy.linalg as la
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.initializers import (
    RandomNormal,
    GlorotUniform,
    GlorotNormal,
    RandomUniform,
)
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import function as Kfunction
from tensorflow.keras.optimizers import SGD
from quantized_network import QuantizedNeuralNetwork
from sys import stdout

N0 = 10 ** 3
N1 = 1
B = 500
w = np.random.rand(N0)
# w = 0.5 * np.ones(N0)

X = np.zeros((B, N0))
X[:, 0] = np.random.randn(B)
X[:, 0] = X[:, 0] / la.norm(X[:, 0])

# # Adversarial orthogonal walk
# u = 0.5 * X[:, 0]
# norms = np.zeros(N0)
# for i in range(1, N0):
#     # Always select a direction which is orthogonal to u.
#     X[:, i] = la.null_space([u])[:, 0]
#     u += 0.5 * X[:, i]
#     norms[i] = la.norm(u)

# Parallel walk.
X = np.zeros((B, N0))
X[0, :] = 1

get_data = (sample for sample in X)
get_data2 = (sample for sample in X)

model = Sequential()
model.add(Dense(N1, activation=None, use_bias=False, input_dim=N0))
model.layers[0].set_weights([w.reshape((N0, 1))])

SD1 = QuantizedNeuralNetwork(network=model, batch_size=B, get_data=get_data, order=1)
SD2 = QuantizedNeuralNetwork(network=model, batch_size=B, get_data=get_data2, order=2)

SD1.quantize_network()
SD2.quantize_network()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(np.arange(N0), la.norm(SD1.residuals[0][0], axis=1))
ax.plot(
    np.arange(N0),
    [
        val
        for pair in zip(
            la.norm(SD2.residuals[0][0], axis=1), la.norm(SD2.residuals[0][0], axis=1)
        )
        for val in pair
    ],
)
ax.legend(["One Step", "One-Two Step"])
ax.set_xlabel(r"$t$", fontsize=14)
ax.set_ylabel(r"$||u_t||$", fontsize=14)
ax.set_title("One Step vs One-Two Step Residuals", fontsize=20)
caption = (
    "Caption: Residuals for the one step and one-two step quantization methods for the one dimensional walk."
    "Weights are initialized to be uniform(0,1). Directions are unit norm. No bias or rectifier were used. The quantization alphabet was ternary."
)
fig.text(0.5, 0.01, caption, ha="center", wrap=True, fontsize=12)

w = w.reshape((N0, 1))
q = SD2.quantized_net.layers[0].get_weights()[0]
la.norm(X @ (w - q), 2)
