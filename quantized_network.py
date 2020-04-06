from numpy import array, zeros, dot, split, cumsum
from numpy.random import permutation, randn
from scipy.linalg import norm
from keras.backend import function as Kfunction
from keras.models import Sequential, clone_model
from keras.layers import Dense
from typing import Optional, Callable
from collections import namedtuple
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes



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

		# Logs the relative error between the data fed through the unquantized network and the
		# quantized network.
		self.layerwise_rel_errs = {	        
					layer_idx: zeros(
								weight_matrix.shape[0], # N_{ell} dimensional feature,
								) 
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

			# Here, we need to loop over the neurons in the previous layer.
			for neuron_idx in range(N_ell):

				wBatch = self.get_batch_data(self.batch_size)
				qBatch = self.get_batch_data(self.batch_size) if self.use_indep_quant_steps else wBatch

				# Remember, neurons correspond to columns in the weight matrix.
				# Only take the output of the neuron_idx neuron.
				wX[:, neuron_idx] = trained_output([wBatch])[0][:, neuron_idx]
				qX[:, neuron_idx] = quant_output([qBatch])[0][:, neuron_idx]

		# Log the relative errors in the data.
		self.layerwise_rel_errs[layer_idx] = [norm(wX[:, t] - qX[:, t])/norm(wX[:,t]) for t in range(N_ell)]

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

	def neuron_dashboard(self, layer_idx: int, neuron_idx: int) -> Axes:

		w = self.trained_net.get_weights()[layer_idx][:,neuron_idx]
		q = self.quantized_net.get_weights()[layer_idx][:, neuron_idx]
		N_ell = w.shape[0]

		fig, axes = subplots(2,2, figsize=(10,10))

		fig.suptitle(f"Dashboard for Neuron {neuron_idx} in Layer {layer_idx}", fontsize=24)

		resid_ax = axes[0,0]
		resid_ax.set_title("Norm of Residuals")
		resid_ax.set_xlabel(r"t", fontsize=14)
		resid_ax.set_ylabel(r"$||u_t||$", fontsize=14)
		resid_ax.plot(range(N_ell), [norm(self.residuals[layer_idx][neuron_idx, t, :]) for t in range(N_ell)], '-o')

		# TODO: What could go in axes[0,1]?

		w_ax = axes[1,0]
		w_ax.set_title("Histogrammed Weights")
		w_ax.hist(w)

		q_ax = axes[1,1]
		q_ax.set_title("Histogrammed Bits")
		q_ax.hist(q)

		return axes

	def layer_dashboard(self, layer_idx: int) -> Axes:

		W = self.trained_net.get_weights()[layer_idx]
		Q = self.quantized_net.get_weights()[layer_idx]
		U_tensor = self.residuals[layer_idx]
		N_ell, N_ell_plus_1 = W.shape

		fig, axes = subplots(2,2, figsize=(10,10))
		fig.suptitle(f"Dashboard for Layer {layer_idx}: $(N_{{L}}, N_{{L+1}}, d)$ = ({N_ell}, {N_ell_plus_1}, {self.batch_size})")
		# adjust the spacing between axes
		fig.tight_layout(pad=6.0)

		# For every t, plot sup_{neurons} ||u_t||
		sup_resids = [max([norm(U_tensor[neuron_idx, t, :]) for neuron_idx in range(N_ell_plus_1)]) 
						for t in range(N_ell)]
		resid_ax = axes[0,0]
		resid_ax.set_title("Supremal Residual Across Neurons")
		resid_ax.set_xlabel("t", fontsize=14)
		resid_ax.set_ylabel(r"$\sup ||u_t||$", fontsize=14)
		resid_ax.plot(range(N_ell), sup_resids, '-o')

		# Plot relative error of data pushed through the quantized net.
		rel_errs_ax = axes[0,1]
		rel_errs_ax.set_title("Relative Data Errors")
		rel_errs_ax.set_xlabel("t", fontsize=14)
		rel_errs_ax.set_ylabel(r"$\frac{||\Delta X||}{||X||}$", fontsize=14)
		rel_errs_ax.plot(range(N_ell), self.layerwise_rel_errs[layer_idx], '-o')

		# Histogram the population of weights as well as the bits.
		w_hist = axes[1,0]
		w_hist.set_title("Histogram of Weights")
		w_hist.hist(W.flatten())

		q_hist = axes[1,1]
		q_hist.set_title("Histogram of Bits")
		q_hist.hist(Q.flatten())

		return axes


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


