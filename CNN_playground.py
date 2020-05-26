import numpy as np
from typing import List, Tuple, Optional
from math import sqrt, log
import scipy.linalg as la
import scipy.stats as stats
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.special import gamma
from itertools import compress
from collections import namedtuple
import logging
from abc import ABC, abstractmethod
from matplotlib.animation import FuncAnimation
from itertools import product
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import RandomUniform
from keras.backend import function as Kfunction
from keras.utils.vis_utils import plot_model
import logging
from sys import stdout
from quantized_network import QuantizedNeuralNetwork
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Import CNN Model
model = ResNet50(weights='imagenet')
# Look at the first Conv2D layer
CNN_layer1 = model.layers[2]
# Look at the shape. The format should be (dim1, dim2, num_channels (RGB), num_filters)
print(CNN_layer1.get_weights()[0].shape)
# Use get_config() on the layer to see what the stride, activation, initializer, and so forth are.
print(CNN_layer1.get_config())

