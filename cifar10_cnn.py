import numpy as np
from math import sqrt, log
import scipy.linalg as la
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.initializers import RandomNormal, GlorotUniform, GlorotNormal, RandomUniform
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import function as Kfunction
from tensorflow.keras.optimizers import SGD
from quantized_network import QuantizedNeuralNetwork, QuantizedCNN
from sys import stdout
%matplotlib osx


logging.basicConfig(stream=stdout)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# Split training from testing
train, test =mnist.load_data()

# Split the training data into two populations. One for training the network and
# one for training the quantization. For now, split it evenly.
train_size = train[0].shape[0]
quant_train_size = 10**4
net_train_size = train_size - quant_train_size
net_train_idxs = np.random.choice(train_size, size=net_train_size, replace=False)
quant_train_idxs = list(
	set(np.arange(train_size)) - set(net_train_idxs)
	)

# Split labels from data. Use one-hot encoding for labels. 
# MNIST ONLY: Reshape images to 28x28x1, because tensorflow is whiny.
X_train, y_train = train
X_test, y_test = test

#MNIST only
train_shape = X_train.shape
test_shape = X_test.shape
X_train = X_train.reshape(train_shape[0], train_shape[1], train_shape[2],1)
X_test = X_test.reshape(test_shape[0], test_shape[1], test_shape[2],1)

num_classes = np.unique(y_train).shape[0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
input_shape = X_train[0].shape

# Futher separate training
X_net_train = X_train[net_train_idxs]
y_net_train = y_train[net_train_idxs]

X_quant_train = X_train[quant_train_idxs]
y_quant_train = y_train[quant_train_idxs]

# Construct a basic convolutional neural network.
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=7, strides=(2,2), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=7,  strides=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Not too many epochs...training CNN's appears to be quite slow.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_net_train, y_net_train, batch_size=128, epochs=20, verbose=True, validation_split=.20)

# Quantize the network.
get_data = (sample for sample in X_quant_train)
# Make it so all data are used.
batch_size = int(np.floor(quant_train_size/(3)))
ignore_layers = [] #[num_layers-1]
is_debug = False
my_quant_net = QuantizedCNN(network=model, batch_size=batch_size, get_data=get_data, logger=logger, is_debug=is_debug)
my_quant_net.quantize_network()

my_quant_net.quantized_net.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
analog_loss, analog_accuracy  = model.evaluate(X_test, y_test, verbose=True)
q_loss, q_accuracy = my_quant_net.quantized_net.evaluate(X_test, y_test, verbose=True)

# Construct MSQ Net.
MSQ_model = clone_model(model)
# Set all the weights to be equal at first. This matters for batch normalization layers.
MSQ_model.set_weights(model.get_weights())
for layer_idx, layer in enumerate(model.layers):
	if layer.__class__.__name__ in ('Dense', 'Conv2D'):
		# Use the same radius as the ternary alphabet in the corresponding layer of the Sigma Delta network.
		rad = max(my_quant_net.quantized_net.layers[layer_idx].get_weights()[0].flatten())
		W, b = model.layers[layer_idx].get_weights()
		Q = np.zeros(W.shape)
		Q[W >= rad/2] = rad
		Q[W <= -rad/2] = -rad
		Q[abs(W) < rad/2] = 0
		MSQ_model.layers[layer_idx].set_weights([Q, b])

MSQ_model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
MSQ_loss, MSQ_accuracy = MSQ_model.evaluate(X_test, y_test, verbose=True)


print(f'Analog Test Loss: {analog_loss:.3}\t\t\tMSQ Test Loss: {MSQ_loss:.3}\t\t\tQuantized Test Loss: {q_loss:.3}')
print(f'Analog Test Accuracy: {analog_accuracy:.3}\t\tMSQ Test Accuracy: {MSQ_accuracy:.3}\t\tQuantized Test Accuracy: {q_accuracy:.3}')
