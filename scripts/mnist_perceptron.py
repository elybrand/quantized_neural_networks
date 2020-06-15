import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.initializers import (
    RandomNormal,
    GlorotUniform,
    GlorotNormal,
    RandomUniform,
)
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from quantized_network import QuantizedNeuralNetwork
from sys import stdout
from itertools import repeat, chain

logging.basicConfig(stream=stdout)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# Load data
train, test = mnist.load_data()

# Separate images from labels
X_train, y_train = train
X_test, y_test = test

# Vectorize the images
img_size = X_train[0].size
X_train = X_train.reshape(X_train.shape[0], img_size)
X_test = X_test.reshape(X_test.shape[0], img_size)

# Convert numerical labels into indicator vectors. There are 10 digits.
num_classes = np.unique(y_train).shape[0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build perceptron. We will vectorize the images.
hidden_layer_sizes = [300, 100]
activation = "relu"
kernel_initializer = GlorotUniform()
model = Sequential()
model.add(
    Dense(
        hidden_layer_sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        use_bias=True,
        input_dim=img_size,
    )
)
model.add(BatchNormalization())
# Add hidden layers
for layer_size in hidden_layer_sizes[1:]:
    model.add(
        Dense(
            layer_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_bias=True,
        )
    )
    model.add(BatchNormalization())
# Add last layer.
model.add(Dense(num_classes, kernel_initializer=kernel_initializer, activation="softmax"))
num_layers = len(model.layers)

# Use SGD. Set momentum to 0.9 if you want to match MATLAB
model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=20,
    verbose=True,
    validation_split=0.20,
)

get_data = (sample for sample in X_train)
for i in range(num_layers-1):
    # Chain together iterators over the entire training set. This is so each layer uses
    # the entire training data.
    get_data = chain(get_data, (sample for sample in X_train))
# get_data2 = (sample for sample in X_train)
batch_size = X_train.shape[0]
alphabet_scalar = 2
ignore_layers = [] 
bits = np.log2(3)
my_quant_net = QuantizedNeuralNetwork(
    network=model,
    batch_size=batch_size,
    get_data=get_data,
    logger=logger,
    ignore_layers=ignore_layers,
    order=1,
    bits=bits,
    alphabet_scalar=alphabet_scalar,
)

# my_quant_net2 = QuantizedNeuralNetwork(
#     network=model,
#     batch_size=batch_size,
#     get_data=get_data2,
#     logger=logger,
#     ignore_layers=ignore_layers,
#     order=2,
#     bits=bits,
#     alphabet_scalar=alphabet_scalar,
# )

my_quant_net.quantize_network()
# my_quant_net2.quantize_network()


# Construct SigmaDelta Net
my_quant_net.quantized_net.compile(
    optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
)
# my_quant_net2.quantized_net.compile(
#     optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
# )
analog_loss, analog_accuracy = model.evaluate(X_test, y_test, verbose=True)
q_loss, q_accuracy = my_quant_net.quantized_net.evaluate(X_test, y_test, verbose=True)
# q2_loss, q2_accuracy = my_quant_net2.quantized_net.evaluate(X_test, y_test, verbose=True)

# Construct MSQ Net.
logger.info("Constructing MSQ net...")
MSQ_model = clone_model(model)
# Set all the weights to be equal at first. This matters for batch normalization layers.
MSQ_model.set_weights(model.get_weights())
for layer_idx, layer in enumerate(model.layers):
    if layer.__class__.__name__ in ("Dense", "Conv2D") and layer_idx not in ignore_layers:
        # Use the same radius as the alphabet in the corresponding layer of the Sigma Delta network.
        rad = max(my_quant_net.quantized_net.layers[layer_idx].get_weights()[0].flatten())
        W, b = model.layers[layer_idx].get_weights()
        m, N = W.shape
        Q = np.array([my_quant_net.bit_round(w, rad) for w in W.flatten()]).reshape(
            W.shape
        )
        MSQ_model.layers[layer_idx].set_weights([Q, b])

MSQ_model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
MSQ_loss, MSQ_accuracy = MSQ_model.evaluate(X_test, y_test, verbose=True)

# See how accuracies degrade as a function of layers quantized. Also calculate compression ratios.
MSQ_metrics = pd.DataFrame(
    {"accuracy": np.zeros(num_layers), "compression": np.zeros(num_layers)},
    index=range(num_layers),
)
SD_metrics = pd.DataFrame(
    {"accuracy": np.zeros(num_layers), "compression": np.zeros(num_layers)},
    index=range(num_layers),
)
# SD2_metrics = pd.DataFrame(
#     {"accuracy": np.zeros(num_layers), "compression": np.zeros(num_layers)},
#     index=range(num_layers),
# )

MSQ_tmp_net = clone_model(model)
MSQ_tmp_net.set_weights(model.get_weights())
SD_tmp_net = clone_model(model)
SD_tmp_net.set_weights(model.get_weights())
# SD2_tmp_net = clone_model(model)
# SD2_tmp_net.set_weights(model.get_weights())

logger.info("Computing layerwise statistics...")
for layer_idx, layer in enumerate(model.layers):
    if layer.__class__.__name__ in ("Dense", "Conv2D") and layer_idx not in ignore_layers:
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

        SD_wts = my_quant_net.quantized_net.layers[layer_idx].get_weights()
        SD_tmp_net.layers[layer_idx].set_weights(SD_wts)
        SD_tmp_net.compile(
            optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        loss, acc = SD_tmp_net.evaluate(X_test, y_test, verbose=False)
        SD_metrics["accuracy"][layer_idx] = acc
        SD_metrics["compression"][layer_idx] = sum(SD_wts[0].flatten() != 0) / supp_W

        # SD2_wts = my_quant_net2.quantized_net.layers[layer_idx].get_weights()
        # SD2_tmp_net.layers[layer_idx].set_weights(SD2_wts)
        # SD2_tmp_net.compile(
        #     optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
        # )
        # loss, acc = SD2_tmp_net.evaluate(X_test, y_test, verbose=False)
        # SD2_metrics["accuracy"][layer_idx] = acc
        # SD2_metrics["compression"][layer_idx] = sum(SD2_wts[0].flatten() != 0) / supp_W
    else:
        MSQ_metrics["accuracy"][layer_idx] = MSQ_metrics["accuracy"][layer_idx - 1]
        MSQ_metrics["compression"][layer_idx] = 1

        SD_metrics["accuracy"][layer_idx] = SD_metrics["accuracy"][layer_idx - 1]
        SD_metrics["compression"][layer_idx] = 1

        # SD2_metrics["accuracy"][layer_idx] = SD2_metrics["accuracy"][layer_idx - 1]
        # SD2_metrics["compression"][layer_idx] = 1

# Plot accuracies and compression ratios
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
axes[0].plot(range(num_layers), MSQ_metrics["accuracy"], "-o")
axes[0].plot(range(num_layers), SD_metrics["accuracy"], "-o")
# axes[0].plot(range(num_layers), SD2_metrics["accuracy"], "-o")
# axes[0].legend(["MSQ", r"$\Sigma\Delta$", r"$\Sigma\Delta 2$"], fontsize=12)
axes[0].legend(["MSQ", r"$\Sigma\Delta$"], fontsize=12)
axes[0].set_title("Test Accuracy vs Layers Quantized", fontsize=22)
axes[0].set_xticks(range(num_layers))
axes[0].set_ylim([0, 1])
axes[0].set_xlabel("Layer Index", fontsize=18)
axes[0].set_ylabel("Test Accuracy", fontsize=18)

width = 0.25
axes[1].bar(np.arange(num_layers) - width, MSQ_metrics["compression"], width)
axes[1].bar(np.arange(num_layers), SD_metrics["compression"], width)
# axes[1].bar(np.arange(num_layers) + width, SD2_metrics["compression"], width)
# axes[1].legend(["MSQ", r"$\Sigma\Delta$", r"$\Sigma\Delta 2$"], fontsize=12)
axes[1].legend(["MSQ", r"$\Sigma\Delta$"], fontsize=12)
axes[1].set_title("Layerwise Weight Compression Ratio", fontsize=22)
axes[1].set_xticks(range(num_layers))
axes[1].set_ylim([0, 1.09])
axes[1].set_xlabel("Layer Index", fontsize=18)
axes[1].set_ylabel("Percentage Weights Zeroed", fontsize=18)
