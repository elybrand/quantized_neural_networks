import numpy as np
from math import sqrt, log
import scipy.linalg as la
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, GlorotUniform, GlorotNormal, RandomUniform
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import function as Kfunction
from tensorflow.keras.optimizers import SGD
from quantized_network import QuantizedNeuralNetwork
from sys import stdout
%matplotlib osx

logging.basicConfig(stream=stdout)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# Load data
train, test = mnist.load_data()

# test_idxs = np.random.choice(test[0].shape[0], 2500, replace=False)

# Split the training data into two populations. One for training the network and
# one for training the quantization. For now, split it evenly.
train_size = train[0].shape[0]
quant_train_size = 10**4
net_train_size = train_size - quant_train_size
net_train_idxs = np.random.choice(train_size, size=net_train_size, replace=False)
quant_train_idxs = list(
	set(np.arange(train_size)) - set(net_train_idxs)
	)

# Separate images from labels
X_train, y_train = train
X_test, y_test = test

# fig, axes = plt.subplots(5, 5)
# axes = axes.ravel()
# for i in range(len(axes)):
# 	axes[i].imshow(X_train[i])
# 	axes[i].axis('off')

# # Normalize pixel values
# X_train = X_train/255
# X_test = X_test/255

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
hidden_layer_sizes = [500, 250, 125]
# hidden_layer_sizes = [150, 200, 100]
activation = 'relu'
kernel_initializer = GlorotUniform()
model = Sequential()
model.add(Dense(hidden_layer_sizes[0], activation=activation, kernel_initializer=kernel_initializer, use_bias=True, input_dim=img_size))
# epsilon=10**(-5) is the MATLAB default.
model.add(BatchNormalization(epsilon=10**(-5)))
# Add hidden layers
for layer_size in hidden_layer_sizes[1:]:
	 model.add(Dense(layer_size, activation=activation, kernel_initializer=kernel_initializer, use_bias=True))
	 model.add(BatchNormalization(epsilon=10**(-5)))
# Add last layer.
model.add(Dense(num_classes, kernel_initializer=kernel_initializer, activation='softmax'))
num_layers = len(model.layers)

# Use SGD. Set momentum to 0.9 if you want to match MATLAB
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_net_train, y_net_train, batch_size=128, epochs=20, verbose=True, validation_split=.20)

# Now quantize the network.
get_data = (sample for sample in X_quant_train)
# Make it so all data are used.
batch_size = int(np.floor(quant_train_size/(len(hidden_layer_sizes)+1)))
ignore_layers = [] #[num_layers-1]
is_debug = False
my_quant_net = QuantizedNeuralNetwork(network=model, batch_size=batch_size, get_data=get_data, logger=logger, ignore_layers=ignore_layers, is_debug=is_debug)
my_quant_net.quantize_network()

# fig, axes = plt.subplots(2, 3, figsize=(25,13))

# axes[0,0].plot(history.history['accuracy'], '-o')
# axes[0,0].plot(history.history['val_accuracy'], '-o')
# axes[0,0].set_title('Model Accuracy', fontsize=18)
# axes[0,0].set_ylabel('Accuracy', fontsize=16)
# axes[0,0].set_xlabel('Epoch',fontsize=16)
# axes[0,0].legend(['training', 'validation'], loc='best')

# # Plot first 3 residuals for first layer
# U0 = my_quant_net.residuals[0][0:3]
# layer0_norms = la.norm(U0, 2, axis=2)
# axes[0,1].plot(layer0_norms.T)
# axes[0,1].set_title("Residual Plots for Layer 0", fontsize=18)
# axes[0,1].set_xlabel(r"$t$", fontsize=16)
# axes[0,1].set_ylabel(r"$||u_t||$", fontsize=16)
# axes[0,1].legend(["Neuron 0", "Neuron 1", "Neuron 2"])

# # Plot first 3 residuals for last layer
# U_last = my_quant_net.residuals[num_layers-1][0:3]
# last_layer_norms = la.norm(U_last, 2, axis=2)
# axes[0,2].plot(last_layer_norms.T)
# axes[0,2].set_title("Residual Plots for Final Layer", fontsize=18)
# axes[0,2].set_xlabel(r"$t$", fontsize=16)
# axes[0,2].set_ylabel(r"$||u_t||$", fontsize=16)
# axes[0,2].legend(["Neuron 0", "Neuron 1", "Neuron 2"])

# # Define functions which will give you the output of the previous hidden layer
# # for both networks.
# L0_trained_output = Kfunction([my_quant_net.trained_net.layers[0].input],
# 										[my_quant_net.trained_net.layers[0].output]
# 								)
# L0_quant_output = Kfunction([my_quant_net.quantized_net.layers[0].input],
# 										[my_quant_net.quantized_net.layers[0].output]
# 								)

# # Histogram the (training set) relative errors across neurons for the first layer.
# train_wX = my_quant_net.layerwise_directions[0]['wX']
# train_qX = my_quant_net.layerwise_directions[0]['qX']
# analog_output = L0_trained_output([train_wX])[0]
# q_output = L0_quant_output([train_qX])[0]
# rel_errs_L0 = np.divide(la.norm(analog_output - q_output, 2, axis=0), la.norm(analog_output, 2, axis=0))
# axes[1,0].hist(rel_errs_L0)
# axes[1,0].set_title("Histogram of First Layer Training Relative Errors", fontsize=18)
# axes[1,0].set_xlabel(r"$\frac{||X(w-q)||_2}{||Xw||_2}$", fontsize=16)

# # Define functions which will give you the output of the previous hidden layer
# # for both networks.
# L_last_trained_output = Kfunction([my_quant_net.trained_net.layers[num_layers-1].input],
# 										[my_quant_net.trained_net.layers[num_layers-1].output]
# 								)
# L_last_quant_output = Kfunction([my_quant_net.quantized_net.layers[num_layers-1].input],
# 										[my_quant_net.quantized_net.layers[num_layers-1].output]
# 								)

# # Histogram the (training set) relative errors across neurons for the last layer.
# train_wX = my_quant_net.layerwise_directions[num_layers-1]['wX']
# train_qX = my_quant_net.layerwise_directions[num_layers-1]['qX']
# analog_output = L_last_trained_output([train_wX])[0]
# q_output = L_last_quant_output([train_qX])[0]
# rel_errs_last = np.divide(la.norm(analog_output - q_output, np.inf, axis=0), la.norm(analog_output, 2, axis=0))
# axes[1,1].hist(rel_errs_last)
# axes[1,1].set_title("Histogram of Last Layer Training Relative Errors", fontsize=18)
# axes[1,1].set_xlabel(r"$\frac{||X(w-q)||_{\infty}}{||Xw||_{\infty}}$", fontsize=16)

# # Histogram the (test set) relative errors across neurons for the last layer.
# W2 = model.layers[2].get_weights()[0].flatten()
# axes[1,2].hist(W2, bins=25)
# axes[1,2].set_title("Histogram of Third Layer Weights", fontsize=18)

# Construct SigmaDelta Net
my_quant_net.quantized_net.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
analog_loss, analog_accuracy  = model.evaluate(X_test, y_test, verbose=True)
q_loss, q_accuracy = my_quant_net.quantized_net.evaluate(X_test, y_test, verbose=True)

# Construct MSQ Net.
MSQ_model = clone_model(model)
# Set all the weights to be equal at first. This matters for batch normalization layers.
MSQ_model.set_weights(model.get_weights())
for layer_idx, layer in enumerate(model.layers):
	if layer.__class__.__name__ == 'Dense':
		# Use the same radius as the ternary alphabet in the corresponding layer of the Sigma Delta network.
		rad = max(my_quant_net.quantized_net.layers[layer_idx].get_weights()[0].flatten())
		W, b = model.layers[layer_idx].get_weights()
		m, N = W.shape
		Q = np.zeros(W.shape)
		Q[W >= rad/2] = rad
		Q[W <= -rad/2] = -rad
		Q[abs(W) < rad/2] = 0
		MSQ_model.layers[layer_idx].set_weights([Q, b])

MSQ_model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
MSQ_loss, MSQ_accuracy = MSQ_model.evaluate(X_test, y_test, verbose=True)

# Plot weight distributions for MSQ layers
fig, axes = plt.subplots(2, len(hidden_layer_sizes)+1, figsize=(23, 7))
axes = axes.ravel()
i = 0
for layer_idx, layer in enumerate(MSQ_model.layers):
	if layer.__class__.__name__ == 'Dense':
		axes[i].hist(layer.get_weights()[0].flatten())
		axes[i].set_title(f"MSQ Weights Layer {layer_idx}")
		i += 1
for layer_idx, layer in enumerate(my_quant_net.quantized_net.layers):
	if layer.__class__.__name__ == 'Dense':
		axes[i].hist(layer.get_weights()[0].flatten())
		axes[i].set_title(f"Sigma Delta Weights Layer {layer_idx}")
		i += 1


# caption = f"Caption: Network with hidden layer widths {hidden_layer_sizes} trained on {train_size} MNIST samples."\
# f" {quant_train_size} samples were used to train the quantization with a batch size of {batch_size} per layer."\
# f" Layers {ignore_layers} were not quantized. Analog and quantized network test accuracies are {analog_accuracy:.3} and {q_accuracy:.3}, respectively."
# fig.text(0.5, 0.01, caption, ha='center', wrap=True, fontsize=12)


print(f'Analog Test Loss: {analog_loss:.3}\t\t\tMSQ Test Loss: {MSQ_loss:.3}\t\t\tQuantized Test Loss: {q_loss:.3}')
print(f'Analog Test Accuracy: {analog_accuracy:.3}\t\t\tMSQ Test Accuracy: {MSQ_accuracy:.3}\t\t\tQuantized Test Accuracy: {q_accuracy:.3}')