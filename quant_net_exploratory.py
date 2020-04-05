from numpy.random import randn
from scipy.linalg import norm
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
%matplotlib osx

if __name__ = "__main__":

	N0 = 10
	N1 = 32
	N2 = 1
	batch_size = 10

	def get_batch_data(batch_size:int):
		# Gaussian data for now.
		return randn(batch_size, N0)

	model = Sequential()
	layer1 = Dense(N1, activation=None, use_bias=False, input_dim=N0)
	layer2 = Dense(N2, activation=None, use_bias=False)
	model.add(layer1)
	model.add(layer2)

	my_quant_net = QuantizedNeuralNetwork(model, batch_size, get_batch_data)
	my_quant_net.quantize_network()

	fig, ax = 

