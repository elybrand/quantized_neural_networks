from numpy import array, zeros, dot, split, cumsum
from numpy.random import permutation, randn
from scipy.linalg import norm
from keras.backend import function as Kfunction
from keras.models import Sequential, Model, clone_model
from keras.layers import Dense
from tensorflow.image import extract_patches
from tensorflow import reshape
from typing import Optional, Callable, List, Generator
from collections import namedtuple
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes

QuantizedNeuron = namedtuple("QuantizedNeuron", ['layer_idx', 'neuron_idx', 'q', 'U'])
QuantizedFilter = namedtuple("QuantizedFilter", ['layer_idx', 'filter_idx', 'channel_idx', 'q_filtr', 'U'])
SegmentedData = namedtuple("SegmentedData", ['wX_seg', 'qX_seg'])

class QuantizedNeuralNetwork():

	def __init__(self, network: Model, batch_size: int, get_data: Generator[array, None, None], logger=None, is_debug=False):
		"""
		CAVEAT: Bias terms are not quantized!
		REMEMBER: TensorFlow flips everything for you. Networks act via

		# TODO: add verbose flag
		# TODO: add functionality to scale weights into [-1, 1].


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

		self.get_data = get_data
		
		# The pre-trained network.
		self.trained_net = network

		# This copies the network structure but not the weights.
		self.quantized_net = clone_model(network) 

		self.batch_size = batch_size

		# Create a dictionary encoding which layers are Dense, and what their dimensions are.
		self.layer_dims = {
						layer_idx: layer.get_weights()[0].shape for layer_idx, layer in enumerate(network.layers) 
							if layer.__class__.__name__ == 'Dense'
					}

		# A dictionary with key being the layer index ell and the values being tensors.
		# self.residuals[ell][neuron_idx, :] is a N_ell x batch_size matrix storing the residuals
		# generated while quantizing the neuron_idx neuron.
		self.residuals = {	        
							layer_idx: zeros((
										dims[1], # N_{ell+1} neurons,
										dims[0], # N_{ell} dimensional feature
										self.batch_size)) 		# Dimension of the residual vectors.
							for layer_idx, dims in self.layer_dims.items()
						}

		# Logs the relative error between the data fed through the unquantized network and the
		# quantized network.
		self.layerwise_rel_errs = {	        
					layer_idx: zeros(
								dims[0], # N_{ell} dimensional feature,
								) 
					for layer_idx, dims in self.layer_dims.items()
				}

		self.logger = logger

		self.is_debug = is_debug

		# This is used to log the directions which are used to choose the bits.
		if self.is_debug:
			self.layerwise_directions = {
				layer_idx: { 'wX': zeros(				
							(self.batch_size, # B vectors in each batch
							dims[0]) # N_ell dimensional feature
							),
							'qX':zeros(				
							(self.batch_size, # B vectors in each batch
							dims[0]) # N_ell dimensional feature
							),
							}
				for layer_idx, dims in self.layer_dims.items()
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
		w = self.trained_net.layers[layer_idx].get_weights()[0][:, neuron_idx]
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

		N_ell, N_ell_plus_1 = self.trained_net.layers[layer_idx].get_weights()[0].shape
		wX = zeros((self.batch_size, N_ell))
		qX = zeros((self.batch_size, N_ell))
		# Placeholder for the weight matrix in the quantized network.
		Q = zeros((N_ell, N_ell_plus_1))
		if layer_idx == 0:
			# Data are assumed to be independent.
			wX = zeros((self.batch_size, N_ell))
			for sample_idx in range(self.batch_size):
				try:
					wX[sample_idx,:] = next(self.get_data)
				except StopIteration:
					# No more samples!
					break
			qX = wX
		else:
			# Define functions which will give you the output of the previous hidden layer
			# for both networks.
			prev_trained_output = Kfunction([self.trained_net.layers[0].input],
										[self.trained_net.layers[layer_idx-1].output]
									)
			prev_quant_output = Kfunction([self.quantized_net.layers[0].input],
									[self.quantized_net.layers[layer_idx-1].output]
									)

			input_size = self.layer_dims[0][0]
			wBatch = zeros((self.batch_size, input_size))
			for sample_idx in range(self.batch_size):
				try:
					wBatch[sample_idx,:] =next(self.get_data)
				except StopIteration:
					# No more samples!
					break
			qBatch = wBatch

			wX = prev_trained_output([wBatch])[0]
			qX = prev_quant_output([qBatch])[0]

		# If you're debugging, log wX and qX.
		if self.is_debug:
			self.layerwise_directions[layer_idx]['wX'] = wX
			self.layerwise_directions[layer_idx]['qX'] = qX

		# Now quantize the neurons. This is parallelizable if you wish to make it so.
		for neuron_idx in range(N_ell_plus_1):

			qNeuron = self.quantize_neuron(layer_idx, neuron_idx, wX, qX)
			if self.logger:
				self.logger.info(f"\tFinished quantizing neuron {neuron_idx} of {N_ell_plus_1}")

			# Update quantized weight matrix and the residual dictionary.
			Q[:, neuron_idx] = qNeuron.q
			self.residuals[layer_idx][neuron_idx,:] = qNeuron.U

		# Update the quantized network. Use the same bias vector as in the analog network for now.
		bias = self.trained_net.layers[layer_idx].get_weights()[1]
		self.quantized_net.layers[layer_idx].set_weights([Q, bias])

		# Log the relative errors in the data incurred by quantizing this layer.
		this_layer_trained_output = Kfunction([self.trained_net.layers[layer_idx].input],
										[self.trained_net.layers[layer_idx].output]
									)
		this_layer_quant_output = Kfunction([self.quantized_net.layers[layer_idx].input],
								[self.quantized_net.layers[layer_idx].output]
								)
		new_wX = this_layer_trained_output([wX])[0]
		new_qX = this_layer_quant_output([qX])[0]
		self.layerwise_rel_errs[layer_idx] = [norm(new_wX[:, t] - new_qX[:, t])/norm(new_wX[:,t]) for t in range(N_ell_plus_1)]

	def quantize_network(self):
		
		# This must be done sequentially.
		for layer_idx, layer in enumerate(self.trained_net.layers):
			# Only quantize dense layers.
			if self.logger:
				self.logger.info(f"Quantizing layer {layer_idx}...")
			if layer.__class__.__name__ == 'Dense':
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

		W = self.trained_net.layers[layer_idx].get_weights()[0]
		Q = self.quantized_net.layers[layer_idx].get_weights()[0]
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
		resid_ax.hist(sup_resids)

		# Plot relative error of data pushed through the quantized net.
		rel_errs_ax = axes[0,1]
		rel_errs_ax.set_title(r"Histogram of $\frac{||\Delta X_t||}{||X_t||}$")
		rel_errs_ax.hist(self.layerwise_rel_errs[layer_idx])

		# Histogram the population of weights as well as the bits.
		w_hist = axes[1,0]
		w_hist.set_title("Histogram of Weights")
		w_hist.hist(W.flatten())

		q_hist = axes[1,1]
		q_hist.set_title("Histogram of Bits")
		q_hist.hist(Q.flatten())

		return axes

class QuantizedCNN(QuantizedNeuralNetwork):

	def __init__(self, network: Model, batch_size: int, get_batch_data: Callable[[int], array], is_debug=False):
		self.get_batch_data = get_batch_data
		
		# The pre-trained network.
		self.trained_net = network

		# This copies the network structure but not the weights.
		self.quantized_net = clone_model(network) 

		# This quantifies how many images are used in a given batch to train a layer. This is subtly different
		# than the batch_size for the perceptron case because the actual data here are *patches* of images.
		self.batch_size = batch_size

		# A dictionary with key being the layer index ell and the values being tensors.
		# self.residuals[ell][neuron_idx, :] is a N_ell x batch_size matrix storing the residuals
		# generated while quantizing the neuron_idx neuron.

		#TODO: this is a pain in the ass, but the batch size in the below dictionary actually needs to be
		# the number of patches, not the number of images.
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
		self.is_debug = is_debug

		# This is used to log the directions which are used to choose the bits.
		if self.is_debug:
			self.layerwise_directions = {
				layer_idx: { 'wX': zeros(				
							(self.batch_size, # B vectors in each batch
							weight_matrix.shape[0]) # N_ell dimensional feature
							),
							'qX':zeros(				
							(self.batch_size, # B vectors in each batch
							weight_matrix.shape[0]) # N_ell dimensional feature
							),
							}
				for layer_idx, weight_matrix in enumerate(network.get_weights())
			}

	def segment_data2D(kernel_size: tuple, strides: tuple, padding, wX: array, qX: array) -> SegmentedData:
		# We need to format these variables so that tensorflow can interpret them
		# for a list of images.
		kernel_sizes_list = [1, kernel_size[0], kernel_size[1], 1]
		strides_list = [1, strides[0], strides[1], 1]
		# Not entirely confident about this rates variable, but let's go with it.
		rates = [1, 1, 1, 1]

		# The output of extract_patches is indexed by (batch, row, column).
		# wX_seg[i,, x, y] is the *flattened* patch for batch i at the xth row stride and the yth column stride.
		# Since we're just dealing with one batch each, and we don't particularly care about the ordering
		# of the patches, we'll just flatten it all into a 2D tensor.
		wX_seg = extract_patches(images=wX,
								sizes=kernel_sizes_list,
								strides=strides_list,
								rates=rates,
								padding=padding)
		qX_seg = extract_patches(images=qX,
								sizes=kernel_sizes_list,
								strides=strides_list,
								rates=rates,
								padding=padding)

		new_shape = (wX_seg.shape[1]*wX_seg.shape[2], wX_seg.shape[3])
		wX_seg = reshape(wX_seg, new_shape).numpy()
		qX_seg = reshape(qX_seg, new_shape).numpy()

		return SegmentedData(wX_seg=wX_seg, qX_seg=qX_seg)

	def quantize_filter2D(self, layer_idx: int, filter_idx: int, wX: array, qX: array) -> List[QuantizedFilter]:

		# Each channel has its own filter, so we need to split by channel. We assume the number of channels 
		# is the last dimension in the tensor.
		num_channels = wX.shape[-1]
		layer = model.layers[layer_idx]
		kernel_size = layer.kernel_size
		strides = layer.strides
		padding = layer.padding
		quantized_filter_list = []
		for channel_idx in range(num_channels):

			# Segment the data into patches.
			channel_wX = wX[:,:,:,channel_idx]
			channel_qX = qX[:,:,:,channel_idx]

			seg_data = segment_data2D(kernel_size, strides, padding, channel_wX, channel_qX)
			channel_wX_patches = seg_data.wX_seg
			channel_qX_patches = seg_data.qX_seg

			filtr = layer.get_weights()[0][:,:,:,filter_idx][:,:,channel_idx]
			# Flatten the filter.
			filtr = reshape(filtr, filtr.size)
			# Now quantize the filter as if it were a neuron in a perceptron, i.e. a column vector.
			# Here, B represents the patch batch (!) size.
			B = seg_data.wX_seg.shape[0]
			u_init = zeros(B)
			q_filtr = zeros(filtr.size)
			U = zeros((filtr.size, B))
			q_filtr[0] = super().quantize_weight(filtr[0], u_init, channel_wX_patches[:,0], channel_qX_patches[:,0])
			U[0,:] = u_init + filtr[0]*channel_wX_patches[:,0] - q_filtr[0]*channel_qX_patches[:,0]

			for t in range(1,filtr.size):
				q_filtr[t] = super().quantize_weight(filtr[t], U[t-1,:], channel_wX_patches[:,t], channel_qX_patches[:,t])
				U[t,:] = U[t-1,:] + w[t]*wX[:,t] - q[t]*qX[:,t]

			quantized_filter_list += [QuantizedFilter(layer_idx=layer_idx, filter_idx=filter_idx, channel_idx=channel_idx, q_filtr=q_filtr, U=U)]

		return quantized_filter_list

	def quantize_conv2D_layer(self, layer_idx: int):
		# wX formatted as an array of images. No flattening.
		num_filters = model.layers[layer_idx].filters
		filter_shape = model.layers[layer_idx].kernel_size
		if layer_idx == 0:
			wX = get_batch_data()
			qX = wX
		else:
			# Define functions which will give you the output of the previous hidden layers
			# for both networks.
			prev_trained_output = Kfunction([self.trained_net.layers[0].input],
										[self.trained_net.layers[layer_idx-1].output]
									)
			prev_quant_output = Kfunction([self.quantized_net.layers[0].input],
									[self.quantized_net.layers[layer_idx-1].output]
									)

			batch = self.get_batch_data(self.batch_size)
			wX = prev_trained_output([batch])[0]
			qX = prev_quant_output([batch])[0]

		# If you're debugging, log wX and qX.
		if self.is_debug:
			self.layerwise_directions[layer_idx]['wX'] = wX
			self.layerwise_directions[layer_idx]['qX'] = qX

		for filter_idx in range(num_filters):
			# This returns a list of quantized filters because for a given patch of an image,
			# each channel has its own convolutional filter. So for a standard RGB channel image,
			# this list should have three elements.
			quantized_filter_list = quantize_filter2D(layer_idx, filter_idx, wX, qX)
			# Now we need to stack all the channel information together again.
			N, B = quantized_filter_list.U.shape
			filter_U = zeros((N, B, num_channels))
			quantized_filter = zeros((filter_shape[0], filter_shape[1], num_channels))
			for channel_filter in quantized_filter_list:
				filter_U[:, :, channel_idx] = channel_filter.U
				quantized_filter[:,:, channel_idx] = channel_filter.q_filtr




	def quantize_network(self):
		# TODO: you need to manually copy the bias terms and all other things that aren't the conv2D layers. Cloning
		# the network only copies the structure.
		pass




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


