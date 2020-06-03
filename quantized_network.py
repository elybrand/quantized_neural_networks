from numpy import array, zeros, dot, split, cumsum, median, nan, reshape
from numpy.random import permutation, randn
from scipy.linalg import norm
from tensorflow.keras.backend import function as Kfunction
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.image import extract_patches
# from tensorflow import reshape
from typing import Optional, Callable, List, Generator
from collections import namedtuple
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from concurrent.futures import ThreadPoolExecutor, as_completed

QuantizedNeuron = namedtuple("QuantizedNeuron", ['layer_idx', 'neuron_idx', 'q', 'U'])
QuantizedFilter = namedtuple("QuantizedFilter", ['layer_idx', 'filter_idx', 'channel_idx', 'q_filtr', 'U'])
SegmentedData = namedtuple("SegmentedData", ['wX_seg', 'qX_seg'])

class QuantizedNeuralNetwork():

	def __init__(self, network: Model, batch_size: int, get_data: Generator[array, None, None], logger=None, is_debug=False, ignore_layers=[]):
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

		# Set all the weights to be the same a priori.
		self.quantized_net.set_weights(network.get_weights())

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

		self.ignore_layers = ignore_layers

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


	def bit_round(self, t: float, rad: float) -> int:
		if abs(t) < (1.0 * rad)/2:
			return 0
		return -rad if t <= (-1.0 * rad)/2 else rad

	def quantize_weight(self, w: float, u: array, X: array, X_tilde: array, rad: float) -> int:
		# This is undefined if X_tilde is zero. In this case, return 0.
		if norm(X_tilde,2) < 10**(-16):
			return 0

		return self.bit_round(dot(X_tilde, u + w*X)/(norm(X_tilde,2)**2), rad)

	def quantize_neuron(self, layer_idx: int, neuron_idx: int, wX: array, qX: array, rad=1) -> QuantizedNeuron:

		N_ell = wX.shape[1]
		u_init = zeros(self.batch_size)
		w = self.trained_net.layers[layer_idx].get_weights()[0][:, neuron_idx]
		q = zeros(N_ell)
		U = zeros((N_ell, self.batch_size))
		# Take columns of the data matrix, since the samples are given via the rows.
		q[0] = self.quantize_weight(w[0], u_init, wX[:,0], qX[:,0], rad)
		U[0,:] = u_init + w[0]*wX[:,0] - q[0]*qX[:,0]

		for t in range(1,N_ell):
			q[t] = self.quantize_weight(w[t], U[t-1,:], wX[:,t], qX[:,t], rad)
			U[t,:] = U[t-1,:] + w[t]*wX[:,t] - q[t]*qX[:,t]

		qNeuron = QuantizedNeuron(layer_idx=layer_idx, neuron_idx=neuron_idx, q=q, U=U)

		return qNeuron

	def quantize_layer(self, layer_idx: int):

		W = self.trained_net.layers[layer_idx].get_weights()[0]
		N_ell, N_ell_plus_1 = W.shape
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

			input_size = self.trained_net.layers[0].get_weights()[0].shape[0]
			wBatch = zeros((self.batch_size, input_size))
			for sample_idx in range(self.batch_size):
				try:
					wBatch[sample_idx,:] = next(self.get_data)
				except StopIteration:
					# No more samples!
					break
			qBatch = wBatch

			# breakpoint()
			wX = prev_trained_output([wBatch])[0]
			qX = prev_quant_output([qBatch])[0]

		# If you're debugging, log wX and qX.
		if self.is_debug:
			self.layerwise_directions[layer_idx]['wX'] = wX
			self.layerwise_directions[layer_idx]['qX'] = qX

		# Set the radius of the ternary alphabet.
		alpha = 1 # This is somewhat arbitrary.
		rad = alpha * median(abs(W.flatten()))

		# Now quantize the neurons.
		with ThreadPoolExecutor() as executor:
			future_to_neuron = {executor.submit(self.quantize_neuron, layer_idx, neuron_idx, wX, qX, rad): neuron_idx for neuron_idx in range(N_ell_plus_1)}
			for future in as_completed(future_to_neuron):
				neuron_idx = future_to_neuron[future]
				try:
					qNeuron = future.result()
				except Exception as exc:
					if self.logger:
						self.logger.error(f"Error quantizing neuron {neuron_idx} in layer {layer_idx}: {exc}")
					else:
						print(f"Error quantizing neuron {neuron_idx} in layer {layer_idx}: {exc}")
				# Update quantized weight matrix and the residual dictionary.
				Q[:, neuron_idx] = qNeuron.q
				self.residuals[layer_idx][neuron_idx,:] = qNeuron.U
				if self.logger:
					self.logger.info(f"\tFinished quantizing neuron {neuron_idx} of {N_ell_plus_1}")

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
		self.layerwise_rel_errs[layer_idx] = [norm(new_wX[:, t] - new_qX[:, t])/norm(new_wX[:,t]) if norm(new_wX[:,t]) > 10**-8 else nan for t in range(N_ell_plus_1)]

	def quantize_network(self):
		
		# This must be done sequentially.
		for layer_idx, layer in enumerate(self.trained_net.layers):
			if layer.__class__.__name__ == 'Dense' and layer_idx not in self.ignore_layers:
				# Only quantize dense layers.
				if self.logger:
					self.logger.info(f"Quantizing layer {layer_idx}...")
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

	#TODO: Add logging option
	#TODO: Add ignore_layer option

	def __init__(self, network: Model, batch_size: int, get_data: Generator[array, None, None], logger=None, is_debug=False):
		self.get_data = get_data
		
		# The pre-trained network.
		self.trained_net = network

		# This copies the network structure but not the weights.
		self.quantized_net = clone_model(network) 
		self.quantized_net.set_weights(network.get_weights())

		# This quantifies how many images are used in a given batch to train a layer. This is subtly different
		# than the batch_size for the perceptron case because the actual data here are *patches* of images.
		self.batch_size = batch_size

		self.logger = logger

	def segment_data2D(self, kernel_size: tuple, strides: tuple, padding, wX: array, qX: array) -> SegmentedData:
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

		# Now at this point, we don't really care about how the patches are grouped. So we reshape them into a 2 tensor, where the patches
		# are vectorized and stored in the rows.
		new_shape = (wX_seg.shape[0]*wX_seg.shape[1]*wX_seg.shape[2], wX_seg.shape[3])
		wX_seg = reshape(wX_seg, new_shape)
		qX_seg = reshape(qX_seg, new_shape)

		return SegmentedData(wX_seg=wX_seg, qX_seg=qX_seg)

	def quantize_filter2D(self, layer_idx: int, filter_idx: int, wX: array, qX: array, rad: float) -> List[QuantizedFilter]:

		# Each channel has its own filter, so we need to split by channel. We assume the number of channels 
		# is the last dimension in the tensor, since this is how Tensorflow formats it (as compared to Theano, which is first).
		num_channels = wX.shape[-1]
		layer = self.trained_net.layers[layer_idx]
		kernel_size = layer.kernel_size
		strides = layer.strides
		padding = layer.padding.upper() # Have to capitalize this apparently...why keras and TF don't coordinate this is beyond me.
		quantized_chan_filter_list = []
		for channel_idx in range(num_channels):

			# Segment the channel data into patches.
			channel_wX = wX[:,:,:,channel_idx]
			channel_qX = qX[:,:,:,channel_idx]

			# We have to reshape into a 4 tensor because Tensorflow is picky.
			channel_wX = reshape(channel_wX, (*channel_wX.shape, 1))
			channel_qX = reshape(channel_qX, (*channel_qX.shape, 1))

			# This will return a tensor where the patches have been flattened as rows.
			seg_data = self.segment_data2D(kernel_size, strides, padding, channel_wX, channel_qX)
			channel_wX_patches = seg_data.wX_seg
			channel_qX_patches = seg_data.qX_seg

			chan_filtr = layer.get_weights()[0][:,:,:,filter_idx][:,:,channel_idx]
			# Flatten the filter.
			chan_filtr = reshape(chan_filtr, chan_filtr.size)
			# Now quantize the filter as if it were a neuron in a perceptron, i.e. a column vector.
			# Here, B represents the patch batch (!) size.
			B = seg_data.wX_seg.shape[0]
			u_init = zeros(B)
			q_filtr = zeros(chan_filtr.size)
			U = zeros((chan_filtr.size, B))
			q_filtr[0] = super().quantize_weight(chan_filtr[0], u_init, channel_wX_patches[:,0], channel_qX_patches[:,0], rad)
			U[0,:] = u_init + chan_filtr[0]*channel_wX_patches[:,0] - q_filtr[0]*channel_qX_patches[:,0]

			for t in range(1,chan_filtr.size):
				q_filtr[t] = super().quantize_weight(chan_filtr[t], U[t-1,:], channel_wX_patches[:,t], channel_qX_patches[:,t], rad)
				U[t,:] = U[t-1,:] + chan_filtr[t]*channel_wX_patches[:,t] - q_filtr[t]*channel_qX_patches[:,t]

			q_filtr = reshape(q_filtr, kernel_size)
			quantized_chan_filter_list += [QuantizedFilter(layer_idx=layer_idx, filter_idx=filter_idx, channel_idx=channel_idx, q_filtr=q_filtr, U=U)]

		return quantized_chan_filter_list

	def quantize_dense_layer(self, layer_idx: int):

		input_shape = self.trained_net.layers[0].input_shape[1:]
		W =  self.trained_net.layers[layer_idx].get_weights()[0]
		Q = zeros(W.shape)
		N_ell_plus_1 = W.shape[1]
		num_channels = W.shape[-2] # The last index is the number of filters in this case.
		batch = zeros((self.batch_size, *input_shape))
		for sample_idx in range(self.batch_size):
			try:
				batch[sample_idx] = next(self.get_data)
			except StopIteration:
				# No more samples!
				break

		if layer_idx == 0:
			wX = batch
			qX = wX
		else:
			# Define functions which will give you the output of the previous hidden layers
			# for both networks. This assumes that the only input this layer receives is from
			# the immediately preceding layer, i.e. no skip layers.
			prev_trained_output = Kfunction([self.trained_net.layers[0].input],
										[self.trained_net.layers[layer_idx-1].output]
									)
			prev_quant_output = Kfunction([self.quantized_net.layers[0].input],
									[self.quantized_net.layers[layer_idx-1].output]
									)

			wX = prev_trained_output([batch])[0]
			qX = prev_quant_output([batch])[0]

		# Set the radius of the ternary alphabet.
		alpha = 1 # This is somewhat arbitrary.
		rad = alpha * median(abs(W.flatten()))

		# Now quantize the neurons.
		with ThreadPoolExecutor() as executor:
			future_to_neuron = {executor.submit(self.quantize_neuron, layer_idx, neuron_idx, wX, qX, rad): neuron_idx for neuron_idx in range(N_ell_plus_1)}
			for future in as_completed(future_to_neuron):
				neuron_idx = future_to_neuron[future]
				try:
					qNeuron = future.result()
				except Exception as exc:
					if self.logger:
						self.logger.error(f"Error quantizing neuron {neuron_idx} in layer {layer_idx}: {exc}")
					else:
						print(f"Error quantizing neuron {neuron_idx} in layer {layer_idx}: {exc}")
				# Update quantized weight matrix.
				Q[:, neuron_idx] = qNeuron.q
				if self.logger:
					self.logger.info(f"\tFinished quantizing neuron {neuron_idx} of {N_ell_plus_1}")

		# Update the quantized network. Use the same bias vector as in the analog network for now.
		bias = self.trained_net.layers[layer_idx].get_weights()[1]
		self.quantized_net.layers[layer_idx].set_weights([Q, bias])

	def quantize_conv2D_layer(self, layer_idx: int):
		# wX formatted as an array of images. No flattening.
		num_filters = self.trained_net.layers[layer_idx].filters
		filter_shape = self.trained_net.layers[layer_idx].kernel_size
		input_shape = self.trained_net.layers[0].input_shape[1:]
		W =  self.trained_net.layers[layer_idx].get_weights()[0]
		num_channels = W.shape[-2] # The last index is the number of filters in this case.
		batch = zeros((self.batch_size, *input_shape))
		for sample_idx in range(self.batch_size):
			try:
				batch[sample_idx] = next(self.get_data)
			except StopIteration:
				# No more samples!
				break

		if layer_idx == 0:
			wX = batch
			qX = wX
		else:
			# Define functions which will give you the output of the previous hidden layers
			# for both networks. This assumes that the only input this layer receives is from
			# the immediately preceding layer, i.e. no skip layers.
			prev_trained_output = Kfunction([self.trained_net.layers[0].input],
										[self.trained_net.layers[layer_idx-1].output]
									)
			prev_quant_output = Kfunction([self.quantized_net.layers[0].input],
									[self.quantized_net.layers[layer_idx-1].output]
									)

			wX = prev_trained_output([batch])[0]
			qX = prev_quant_output([batch])[0]

		# Set the radius of the ternary alphabet.
		alpha = 1 # This is somewhat arbitrary.
		rad = alpha * median(abs(W.flatten()))

		Q = zeros(W.shape)
		# # This should be multi-threaded.
		# for filter_idx in range(num_filters):
		# 	# This returns a list of quantized channel filters because for a given patch of an image,
		# 	# each channel has its own convolutional filter. So for a standard RGB channel image,
		# 	# this list should have three elements.
		# 	quantized_chan_filter_list = self.quantize_filter2D(layer_idx, filter_idx, wX, qX, rad)
		# 	# Now we need to stack all the channel information together again.
		# 	N, B = quantized_chan_filter_list[0].U.shape
		# 	filter_U = zeros((N, B, num_channels))
		# 	# Again, following Tensorflow convention that the channel information is the last component.
		# 	quantized_filter = zeros((filter_shape[0], filter_shape[1], num_channels))
		# 	for channel_filter in quantized_chan_filter_list:
		# 		channel_idx = channel_filter.channel_idx
		# 		filter_U[:, :, channel_idx] = channel_filter.U
		# 		quantized_filter[:,:, channel_idx] = channel_filter.q_filtr

		# 	Q[:,:,:,filter_idx] = quantized_filter

		# 	if self.logger:
		# 		self.logger.info(f"\tFinished quantizing filter {filter_idx} of {num_filters}")

		# Now quantize the neurons.
		with ThreadPoolExecutor() as executor:
			future_to_filter = {executor.submit(self.quantize_filter2D, layer_idx, filter_idx, wX, qX, rad): filter_idx for filter_idx in range(num_filters)}
			for future in as_completed(future_to_filter):
				filter_idx = future_to_filter[future]
				try:
					quantized_chan_filter_list = future.result()
				except Exception as exc:
					if self.logger:
						self.logger.error(f"Error quantizing filter {filter_idx} in layer {layer_idx}: {exc}")
					else:
						print(f"Error quantizing neuron {filter_idx} in layer {layer_idx}: {exc}")
						
				# Now we need to stack all the channel information together again.
				N, B = quantized_chan_filter_list[0].U.shape
				filter_U = zeros((N, B, num_channels))
				# Again, following Tensorflow convention that the channel information is the last component.
				quantized_filter = zeros((filter_shape[0], filter_shape[1], num_channels))
				for channel_filter in quantized_chan_filter_list:
					channel_idx = channel_filter.channel_idx
					filter_U[:, :, channel_idx] = channel_filter.U
					quantized_filter[:,:, channel_idx] = channel_filter.q_filtr

				Q[:,:,:,filter_idx] = quantized_filter

				if self.logger:
					self.logger.info(f"\tFinished quantizing filter {filter_idx} of {num_filters}")

		# Now update the layer in the quantized network. Leave the bias the same for now.
		b = self.quantized_net.layers[layer_idx].get_weights()[1]
		self.quantized_net.layers[layer_idx].set_weights([Q, b])


	def quantize_network(self):
		
		for layer_idx, layer in enumerate(self.trained_net.layers):
			if layer.__class__.__name__ == 'Dense':
				# Use parent class quantize layer
				if self.logger:
					self.logger.info(f"Quanziting (dense) layer {layer_idx}...")
				self.quantize_dense_layer(layer_idx)
			if layer.__class__.__name__ == 'Conv2D':
				if self.logger:
					self.logger.info(f"Quanziting (Conv2D) layer {layer_idx}...")
				self.quantize_conv2D_layer(layer_idx)


