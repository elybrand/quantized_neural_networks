from numpy import array, zeros, dot, split, cumsum
from numpy.random import permutation, randn
from scipy.linalg import norm
from keras.backend import function as Kfunction
from keras.models import Sequential, clone_model
from keras.layers import Dense
from typing import Optional, Callable
from collections import namedtuple

QuantizedNeuron = namedtuple("QuantizedNeuron", ['layer_idx', 'neuron_idx', 'q', 'U'])

class QuantizedNeuralNetwork():

	def __init__(self, network: Sequential, batch_size: int, get_batch_data: Callable[[int], array], use_indep_quant_steps=True):
		"""
		CAVEAT: No bias terms for now!
		REMEMBER: TensorFlow flips everything for you. Networks act via

		batch_size x N_ell 	   N_ell x N_{ell+1} 

		[ -- X_1^T -- ] 	[  |	 |		 |	]	
		[ -- X_2^T -- ] 	[ w_1	w_2		w_3	] 
		[	  .   	  ] 	[  |	 |		 |	]	
		[	  .		  ]
		[	  .		  ] 
		[ -- X_B^T -- ] 

		That means our residual matrix for quantizing the j^th neuron (i.e. w_j as detailed above) will be of the form

			N_ell x batch_size

		[ -- u1^T -- 	  ]
		[	  .		 	  ]
		[	  .		 	  ]
		[	  .		 	  ]
		[ -- u_N_ell^T -- ]
		"""

		self.get_batch_data = get_batch_data
		
		# The pre-trained network.
		self.trained_net = network

		# This copies the network structure but not the weights.
		self.quantized_net = clone_model(network) 

		self.batch_size = batch_size

		self.use_indep_quant_steps = use_indep_quant_steps

		# A dictionary with key being the layer index ell and the values being tensors.
		# self.residuals[ell][neuron_idx, :] is a N_ell x batch_size matrix storing the residuals
		# generated while quantizing the neuron_idx neuron.
		self.residuals = {	        
							layer_idx: zeros((
										weight_matrix.shape[1], # N_{ell+1} neurons,
										weight_matrix.shape[0], # N_{ell} dimensional feature
										self.batch_size)) 		# Dimension of the residual vectors.
							for layer_idx, weight_matrix in enumerate(network.get_weights())
						}

		# Logs the perterbations in the directions caused by passing the data through the unquantized weights
		# and through the quantized weights. For layer ell, we'll store these vectors as rows in a N_ell
		# by batch_size matrix.
		self.step_variations = {	        
					layer_idx: zeros((
								weight_matrix.shape[0], # N_{ell} dimensional feature,
								self.batch_size)) 		# Dimension of the variations in directions.
					for layer_idx, weight_matrix in enumerate(network.get_weights())
				}

	def bit_round(self, t: float) -> int:
		if abs(t) < 1/2:
			return 0
		return -1 if t <= -1/2 else 1

	def quantize_weight(self, w: float, u: array, X: array, X_tilde: array) -> int:
		# This is undefined if X_tilde is zero. In this case, return 0.
		if norm(X_tilde,2) < 10**(-16):
			return 0

		return self.bit_round(dot(X_tilde, u + w*X)/(norm(X_tilde,2)**2))

	def quantize_neuron(self, layer_idx: int, neuron_idx: int, wX: array, qX: array) -> QuantizedNeuron:

		N_ell = wX.shape[1]
		u_init = zeros(self.batch_size)
		w = self.trained_net.get_weights()[layer_idx][:, neuron_idx]
		q = zeros(N_ell)
		U = zeros((N_ell, self.batch_size))
		# Take columns of the data matrix, since the samples are given via the rows.
		q[0] = self.quantize_weight(w[0], u_init, wX[:,0], qX[:,0])
		U[0,:] = u_init + w[0]*wX[:,0] - q[0]*qX[:,0]

		for t in range(1,N_ell):
			q[t] = self.quantize_weight(w[t], U[t-1,:], wX[:,t], qX[:,t])
			U[t,:] = U[t-1,:] + w[t]*wX[:,t] - q[t]*qX[:,t]

		qNeuron = QuantizedNeuron(layer_idx=layer_idx, neuron_idx=neuron_idx, q=q, U=U)

		return qNeuron

	def quantize_layer(self, layer_idx: int):
		
		# Generate independent steps using the batching procedure.
		N_ell, N_ell_plus_1 = self.trained_net.get_weights()[layer_idx].shape
		wX = zeros((self.batch_size, N_ell))
		qX = zeros((self.batch_size, N_ell))
		# Placeholder for the weight matrix in the quantized network.
		Q = zeros((N_ell, N_ell_plus_1))
		if layer_idx == 0:
			# Data are assumed to be independent.
			wX = self.get_batch_data(self.batch_size)
			qX = wX
		else:
			# Define functions which will give you the output of the previous hidden layers
			# for both networks.
			trained_output = Kfunction([self.trained_net.layers[0].input],
										[self.trained_net.layers[layer_idx-1].output]
									)
			quant_output = Kfunction([self.quantized_net.layers[0].input],
									[self.quantized_net.layers[layer_idx-1].output]
									)

			for neuron_idx in range(N_ell_plus_1):

				wBatch = self.get_batch_data(self.batch_size)
				qBatch = self.get_batch_data(self.batch_size) if self.use_indep_quant_steps else wBatch

				# Remember, neurons correspond to columns in the weight matrix.
				# Only take the output of the neuron_idx neuron.
				wX[:, neuron_idx] = trained_output([wBatch])[0][:, neuron_idx]
				qX[:, neuron_idx] = quant_output([qBatch])[0][:, neuron_idx]

		# Log the Delta X_t's.
		self.step_variations[layer_idx] = (wX - qX).T

		# Now quantize the neurons. This is parallelizable if you wish to make it so.
		for neuron_idx in range(N_ell_plus_1):

			qNeuron = self.quantize_neuron(layer_idx, neuron_idx, wX, qX)

			# Update quantized weight matrix and the residual dictionary.
			Q[:, neuron_idx] = qNeuron.q
			self.residuals[layer_idx][neuron_idx,:] = qNeuron.U

		# Update the quantized network.
		self.quantized_net.layers[layer_idx].set_weights([Q])

	def quantize_network(self):
		
		# This must be done sequentially.
		for layer_idx in range(len(self.trained_net.layers)):
			self.quantize_layer(layer_idx)

if __name__ == "__main__":

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


