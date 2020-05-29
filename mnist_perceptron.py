import numpy as np
from math import sqrt, log
import scipy.linalg as la
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedNeuralNetwork
from sys import stdout
%matplotlib osx

logging.basicConfig(stream=stdout)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# Load MNIST data
train, test = mnist.load_data(path="mnist.npz")

# Split the training data into two populations. One for training the network and
# one for training the quantization. For now, split it evenly.
train_size = train[0].shape[0]
net_train_size = int(0.3*train_size)
quant_train_size = train_size - net_train_size
net_train_idxs = np.random.choice(train_size, size=net_train_size, replace=False)
quant_train_idxs = list(
	set(np.arange(train_size)) - set(net_train_idxs)
	)

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

# Futher separate training
X_net_train = X_train[net_train_idxs]
y_net_train = y_train[net_train_idxs]

X_quant_train = X_train[quant_train_idxs]
y_quant_train = y_train[quant_train_idxs]

# Build perceptron. We will vectorize the images.
hidden_layer_sizes = [32, 25]
activation = 'sigmoid'
model = Sequential()
model.add(Dense(hidden_layer_sizes[0], activation='sigmoid', use_bias=True, input_dim=img_size))
# Add hidden layers
for layer_size in hidden_layer_sizes[1:]:
	 model.add(Dense(layer_size, activation=activation, use_bias=True))
	 # Batch normalization?
# Add last layer.
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_net_train, y_net_train, batch_size=128, epochs=100, verbose=True, validation_split=.20)


# Now quantize the network.
get_data = (sample for sample in X_quant_train)
# Make it so all data are used.
batch_size = int(np.floor(quant_train_size/(len(hidden_layer_sizes)+1)))
my_quant_net = QuantizedNeuralNetwork(network=model, batch_size=batch_size, get_data=get_data, logger=logger)
my_quant_net.quantize_network()

fig, axes = plt.subplots(2, 3, figsize=(25,13))

axes[0,0].plot(history.history['accuracy'], '-o')
axes[0,0].plot(history.history['val_accuracy'], '-o')
axes[0,0].set_title('Model Accuracy', fontsize=18)
axes[0,0].set_ylabel('Accuracy', fontsize=16)
axes[0,0].set_xlabel('Epoch',fontsize=16)
axes[0,0].legend(['training', 'validation'], loc='best')

# Plot first 3 residuals for first layer
U0 = my_quant_net.residuals[0][0:3]
layer0_norms = la.norm(U0, 2, axis=2)
axes[0,1].plot(layer0_norms.T)
axes[0,1].set_title("Residual Plots for Layer 0", fontsize=18)
axes[0,1].set_xlabel(r"$t$", fontsize=16)
axes[0,1].set_ylabel(r"$||u_t||$", fontsize=16)
axes[0,1].legend(["Neuron 0", "Neuron 1", "Neuron 2"])

# Plot first 3 residuals for last layer
U_last = my_quant_net.residuals[len(model.layers)-1][0:3]
last_layer_norms = la.norm(U_last, 2, axis=2)
axes[0,2].plot(last_layer_norms.T)
axes[0,2].set_title("Residual Plots for Final Layer", fontsize=18)
axes[0,2].set_xlabel(r"$t$", fontsize=16)
axes[0,2].set_ylabel(r"$||u_t||$", fontsize=16)
axes[0,2].legend(["Neuron 0", "Neuron 1", "Neuron 2"])

# Histogram the layers weights.
W0 = model.layers[0].get_weights()[0].flatten()
axes[1,0].hist(W0, bins=100)
axes[1,0].set_title("Histogram of First Layer Weights", fontsize=18)

W1 = model.layers[1].get_weights()[0].flatten()
axes[1,1].hist(W1, bins=100)
axes[1,1].set_title("Histogram of Second Layer Weights", fontsize=18)


W2 = model.layers[2].get_weights()[0].flatten()
axes[1,2].hist(W2, bins=25)
axes[1,2].set_title("Histogram of Third Layer Weights", fontsize=18)

my_quant_net.quantized_net.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
analog_loss, analog_accuracy  = model.evaluate(X_test, y_test, verbose=True)
q_loss, q_accuracy = my_quant_net.quantized_net.evaluate(X_test, y_test, verbose=True)

print(f'Analog Test Loss: {analog_loss:.3}\t\tQuantized Test Loss:{q_loss:.3}')
print(f'Analog Test Accuracy: {analog_accuracy:.3}\tQuantized Test Accuracy: {q_accuracy:.3}')