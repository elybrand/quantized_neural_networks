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
from quantized_network import QuantizedNeuralNetwork
from sys import stdout
%matplotlib osx


logging.basicConfig(stream=stdout)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# Split training from testing
train, test =mnist.load_data()

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

# Construct a basic convolutional neural network.
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=3,  strides=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Not too many epochs...training CNN's appears to be quite slow.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=3, verbose=True, validation_split=.20)
loss, accuracy  = model.evaluate(X_test, y_test, verbose=True)
