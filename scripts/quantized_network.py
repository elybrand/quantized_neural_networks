from numpy import (
    array,
    zeros,
    dot,
    median,
    nan,
    reshape,
    log2,
    linspace,
    argmin,
    abs,
    delete,
    inf,
    argmax,
)
from scipy.linalg import norm
from tensorflow.keras.backend import function as Kfunction
from tensorflow.keras.models import Model, clone_model
from tensorflow.image import extract_patches
from typing import List, Generator
from collections import namedtuple
from itertools import product
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from time import time

QuantizedNeuron = namedtuple("QuantizedNeuron", ["layer_idx", "neuron_idx", "q"])
QuantizedFilter = namedtuple(
    "QuantizedFilter", ["layer_idx", "filter_idx", "channel_idx", "q_filtr"]
)
SegmentedData = namedtuple("SegmentedData", ["wX_seg", "qX_seg"])


class QuantizedNeuralNetwork:
    def __init__(
        self,
        network: Model,
        batch_size: int,
        get_data: Generator[array, None, None],
        logger=None,
        ignore_layers=[],
        bits=log2(3),
        alphabet_scalar=1,
    ):

        self.get_data = get_data

        # The pre-trained network.
        self.trained_net = network

        # This copies the network structure but not the weights.
        self.quantized_net = clone_model(network)

        # Set all the weights to be the same a priori.
        self.quantized_net.set_weights(network.get_weights())

        self.batch_size = batch_size

        self.alphabet_scalar = alphabet_scalar

        # Create a dictionary encoding which layers are Dense, and what their dimensions are.
        self.layer_dims = {
            layer_idx: layer.get_weights()[0].shape
            for layer_idx, layer in enumerate(network.layers)
            if layer.__class__.__name__ == "Dense"
        }

        # This determines the alphabet. There will be 2**bits atoms in our alphabet.
        self.bits = bits

        # Construct the (unscaled) alphabet. Layers will scale this alphabet based on the
        # distribution of that layer's weights.
        self.alphabet = linspace(-1, 1, num=int(round(2 ** (bits))))

        self.logger = logger

        self.ignore_layers = ignore_layers

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _bit_round(self, t: float, rad: float) -> float:
        """Rounds a quantity to the nearest atom in the (scaled) quantization alphabet.

        Parameters
        -----------
        t : float
            The value to quantize.
        rad : float
            Scaling factor for the quantization alphabet.

        Returns
        -------
        bit : float
            The quantized value.
        """

        # Scale the alphabet appropriately.
        layer_alphabet = rad * self.alphabet
        return layer_alphabet[argmin(abs(layer_alphabet - t))]

    def _quantize_weight(
        self, w: float, u: array, X: array, X_tilde: array, rad: float
    ) -> float:
        """Quantizes a single weight of a neuron.

        Parameters
        -----------
        w : float
            The weight.
        u : array ,
            Residual vector.
        X : array
            Vector from the analog network's random walk.
        X_tilde : array
            Vector from the quantized network's random walk.
        rad : float
            Scaling factor for the quantization alphabet.

        Returns
        -------
        bit : float
            The quantized value.
        """

        if norm(X_tilde, 2) < 10 ** (-16):
            return 0

        if abs(dot(X_tilde, u)) < 10 ** (-10):
            return self._bit_round(w, rad)

        return self._bit_round(dot(X_tilde, u + w * X) / (norm(X_tilde, 2) ** 2), rad)

    def _quantize_neuron(
        self,
        layer_idx: int,
        neuron_idx: int,
        wX: array,
        qX: array,
        rad=1,
    ) -> QuantizedNeuron:
        """Quantizes a single neuron in a Dense layer.

        Parameters
        -----------
        layer_idx : int
            Index of the Dense layer.
        neuron_idx : int,
            Index of the neuron in the Dense layer.
        wX : array
            Layer input for the analog convolutional neural network.
        qX : array
            Layer input for the quantized convolutional neural network.
        rad : float
            Scaling factor for the quantization alphabet.

        Returns
        -------
        QuantizedNeuron: NamedTuple
            A tuple with the layer and neuron index, as well as the quantized neuron.
        """

        N_ell = wX.shape[1]
        u = zeros(self.batch_size)
        w = self.trained_net.layers[layer_idx].get_weights()[0][:, neuron_idx]
        q = zeros(N_ell)
        for t in range(N_ell):
            q[t] = self._quantize_weight(w[t], u, wX[:, t], qX[:, t], rad)
            u += w[t] * wX[:, t] - q[t] * qX[:, t]

        return QuantizedNeuron(layer_idx=layer_idx, neuron_idx=neuron_idx, q=q)

    def _get_layer_data(self, layer_idx: int):
        """Gets the input data for the layer at a given index.

        Parameters
        -----------
        layer_idx : int
            Index of the layer.

        Returns
        -------
        tuple: (array, array)
            A tuple of arrays, with the first entry being the input for the analog network
            and the latter being the input for the quantized network.
        """

        layer = self.trained_net.layers[layer_idx]
        layer_data_shape = layer.input_shape[1:] if layer.input_shape[0] is None else layer.input_shape
        wX = zeros((self.batch_size, *layer_data_shape))
        qX = zeros((self.batch_size, *layer_data_shape))
        if layer_idx == 0:
            for sample_idx in range(self.batch_size):
                try:
                    wX[sample_idx, :] = next(self.get_data)
                except StopIteration:
                    # No more samples!
                    break
            qX = wX
        else:
            # Define functions which will give you the output of the previous hidden layer
            # for both networks.
            prev_trained_output = Kfunction(
                [self.trained_net.layers[0].input],
                [self.trained_net.layers[layer_idx - 1].output],
            )
            prev_quant_output = Kfunction(
                [self.quantized_net.layers[0].input],
                [self.quantized_net.layers[layer_idx - 1].output],
            )
            input_layer = self.trained_net.layers[0]
            input_shape = input_layer.input_shape[1:] if input_layer.input_shape[0] is None else input_layer.input_shape
            batch = zeros((self.batch_size, *input_shape))
            for sample_idx in range(self.batch_size):
                try:
                    batch[sample_idx, :] = next(self.get_data)
                except StopIteration:
                    # No more samples!
                    break

            wX = prev_trained_output([batch])[0]
            qX = prev_quant_output([batch])[0]

        return (wX, qX)

    def _update_weights(self, layer_idx: int, Q: array):
        """Updates the weights of the quantized neural network given a layer index and
        quantized weights.

        Parameters
        -----------
        layer_idx : int
            Index of the Conv2D layer.
        Q : array
            The quantized weights.
        """

        # Update the quantized network. Use the same bias vector as in the analog network for now.
        if self.trained_net.layers[layer_idx].use_bias:
            bias = self.trained_net.layers[layer_idx].get_weights()[1]
            self.quantized_net.layers[layer_idx].set_weights([Q, bias])
        else:
            self.quantized_net.layers[layer_idx].set_weights([Q])

    def _quantize_layer(self, layer_idx: int):
        """Quantizes a Dense layer of a multi-layer perceptron.

        Parameters
        -----------
        layer_idx : int
            Index of the Dense layer.
        """

        W = self.trained_net.layers[layer_idx].get_weights()[0]
        N_ell, N_ell_plus_1 = W.shape
        # Placeholder for the weight matrix in the quantized network.
        Q = zeros(W.shape)
        N_ell_plus_1 = W.shape[1]
        wX, qX = self._get_layer_data(layer_idx)

        # Set the radius of the alphabet.
        rad = self.alphabet_scalar * median(abs(W.flatten()))

        for neuron_idx in range(N_ell_plus_1):
            self._log(f"\tQuantizing neuron {neuron_idx} of {N_ell_plus_1}...")
            tic = time()
            qNeuron = self._quantize_neuron(layer_idx, neuron_idx, wX, qX, rad)
            Q[:, neuron_idx] = qNeuron.q

            self._log(f"\tdone. {time() - tic :.2f} seconds.")

            self._update_weights(layer_idx, Q)

    def quantize_network(self):
        """Quantizes all Dense layers that are not specified by the list of ignored layers."""

        # This must be done sequentially.
        for layer_idx, layer in enumerate(self.trained_net.layers):
            if (
                layer.__class__.__name__ == "Dense"
                and layer_idx not in self.ignore_layers
            ):
                # Only quantize dense layers.
                self._log(f"Quantizing layer {layer_idx}...")

                self._quantize_layer(layer_idx)

                self._log(f"done. {layer_idx}...")


class QuantizedCNN(QuantizedNeuralNetwork):

    def __init__(
        self,
        network: Model,
        batch_size: int,
        get_data: Generator[array, None, None],
        logger=None,
        bits=log2(3),
        alphabet_scalar=1,
    ):

        self.get_data = get_data
        self.trained_net = network
        self.quantized_net = clone_model(network)
        self.quantized_net.set_weights(network.get_weights())

        # This quantifies how many images are used in a given batch to train a layer. This is subtly different
        # than the batch_size for the perceptron case because the actual data here are *patches* of images.
        self.batch_size = batch_size

        self.alphabet_scalar = alphabet_scalar
        self.bits = bits
        self.alphabet = linspace(-1, 1, num=int(round(2 ** (bits))))
        self.logger = logger

    def _segment_data2D(
        self, kernel_size: tuple, strides: tuple, padding: str, wX: array, qX: array
    ) -> SegmentedData:
        """Reshapes image tensor data into a 2D tensor. The rows of this 2D tensor are
        the flattened patches which are the arguments of the convolutions in a Conv2D layer.

        Parameters
        -----------
        kernel_size : (int, int)
            Shape of the kernel for a Conv2D layer.
        strides : (int, int)
            Strides of the kernel for a Conv2D layer.
        padding : string
            Either 'valid' or 'same', as in docstring for Conv2D layer.
        wX : 2D array
            Layer (channel) input for the analog convolutional neural network.
        qX : 2D array
            Layer (channel) input for the quantized convolutional neural network.

        Returns
        -------
        SegmentedData: NamedTuple(wX_seg, qX_seg)
            Both wX_seg and qX_seg are 2D tensors whose rows are the flattened patches
            used in the convolutions of the analog and quantized Conv2D layers, respectively.
        """

        kernel_sizes_list = [1, kernel_size[0], kernel_size[1], 1]
        strides_list = [1, strides[0], strides[1], 1]
        rates = [1, 1, 1, 1]

        wX_seg = extract_patches(
            images=wX,
            sizes=kernel_sizes_list,
            strides=strides_list,
            rates=rates,
            padding=padding,
        )
        qX_seg = extract_patches(
            images=qX,
            sizes=kernel_sizes_list,
            strides=strides_list,
            rates=rates,
            padding=padding,
        )

        # Reshape tensor data into a 2 tensor, where the patches are vectorized and stored in the rows.
        new_shape = (wX_seg.shape[0] * wX_seg.shape[1] * wX_seg.shape[2], wX_seg.shape[3])
        wX_seg = reshape(wX_seg, new_shape)
        qX_seg = reshape(qX_seg, new_shape)

        return SegmentedData(wX_seg=wX_seg, qX_seg=qX_seg)

    def _quantize_channel(
        self,
        layer_idx: int,
        filter_idx: int,
        channel_idx: int,
        wX: array,
        qX: array,
        rad=1,
    ) -> QuantizedFilter:
        """Quantizes a single channel filter in a Conv2D layer.

        Parameters
        -----------
        layer_idx : int
            Index of the Conv2D layer.
        filter_idx : int,
            Index of the neuron in the Conv2D layer.
        channel_idx : int
            Index of the channel in the Conv2D layer.
        wX : 2D array
            Layer (channel) input for the analog convolutional neural network, where 
            the rows are vectorized patches. 
        qX : 2D array
            Layer (channel) input for the quantized convolutional neural network, where 
            the rows are vectorized patches.
        rad : float
            Scaling factor for the quantization alphabet.

        Returns
        -------
        QuantizedFilter: NamedTuple
            A tuple with the layer, filter, and channel index, as well as the quantized channel.
        """

        B = wX.shape[0]
        u = zeros(B)
        chan_filtr = self.trained_net.layers[layer_idx].get_weights()[0][:, :, :, filter_idx][:, :, channel_idx]
        filter_shape = chan_filtr.shape
        chan_filtr = reshape(chan_filtr, chan_filtr.size)
        q_filtr = zeros(chan_filtr.size)

        for t in range(chan_filtr.size):
            q_filtr[t] = super()._quantize_weight(
                chan_filtr[t],
                u,
                wX[:, t],
                qX[:, t],
                rad,
            )
            u += chan_filtr[t] * wX[:, t] - q_filtr[t] * qX[:, t]

        q_filtr = reshape(q_filtr, filter_shape)

        return QuantizedFilter(
                    layer_idx=layer_idx,
                    filter_idx=filter_idx,
                    channel_idx=channel_idx,
                    q_filtr=q_filtr,
                )

    def _quantize_filter2D(
        self, layer_idx: int, filter_idx: int, wX_patch_tensor: array, qX_patch_tensor: array, rad: float
    ) -> List[QuantizedFilter]:
        """Quantizes a given filter, or kernel, in a Conv2D layer by quantizing each channel filter
        as though it were a neuron in a perceptron.

        Parameters
        -----------
        layer_idx : int
            Index of the Conv2D layer.
        filter_idx : int
            Index of the filter in the Conv2D layer.
        wX_patch_tensor : 3D array
            Patch tensor as in the return of _get_patch_tensors
        qX_patch_tensor : array
            Patch tensor as in the return of _get_patch_tensors

        Returns
        -------
        quantized_chan_filter_list: List[QuantizedFilter]
            Returns a list of quantized channel filters.
        """
        num_channels = wX_patch_tensor.shape[-1]
        quantized_chan_filter_list =[
            self._quantize_channel(
                layer_idx, 
                filter_idx, 
                channel_idx, 
                wX_patch_tensor[:,:,channel_idx], 
                qX_patch_tensor[:,:,channel_idx], 
                rad,
            )
            for channel_idx in range(num_channels)
            ]

        return quantized_chan_filter_list

    def _quantize_dense_layer(self, layer_idx: int):

        super()._quantize_layer(layer_idx)

    def _get_patch_tensors(self, kernel_size: tuple, strides: tuple, padding: str, wX: array, qX: array) -> tuple:
        """Returns a 3D tensor whose first two axes encode the 2D tensor of vectorized
        patches of images, and the last axis encodes the channel.

        Parameters
        -----------
        kernel_size: tuple
            Tuple of integers encoding the dimensions of the filter/kernel
        strides: tuple
            Tuple of integers encoding the stride information of the filter/kernel
        padding: string
            Padding argument for Conv2D layer.
        wX : array
            Layer input for the analog convolutional neural network.
        qX : array
            Layer input for the quantized convolutional neural network.

        Returns
        -------
        (wX_patch_tensor, qX_patch_tensor): tuple(array, array) 
            The two patch tensors for the analog and quantized networks, respectively.
        """
        num_channels = wX[0].shape[-1]
        wX_patch_tensor = None
        qX_patch_tensor = None
        for channel_idx in range(num_channels):
            channel_wX = wX[:, :, :, channel_idx]
            channel_qX = qX[:, :, :, channel_idx]

            # We have to reshape into a 4 tensor because Tensorflow is picky.
            channel_wX = reshape(channel_wX, (*channel_wX.shape, 1))
            channel_qX = reshape(channel_qX, (*channel_qX.shape, 1))

            seg_data = self._segment_data2D(
                kernel_size, strides, padding, channel_wX, channel_qX
            )

            if wX_patch_tensor is None:
                wX_patch_tensor = zeros((*seg_data[0].shape, num_channels))
                qX_patch_tensor = zeros(wX_patch_tensor.shape)

            wX_patch_tensor[:, :, channel_idx] = seg_data.wX_seg
            qX_patch_tensor[:, :, channel_idx] = seg_data.qX_seg

        return (wX_patch_tensor, qX_patch_tensor)

    def _quantize_conv2D_layer(self, layer_idx: int):
        # wX formatted as an array of images. No flattening.
        layer = self.trained_net.layers[layer_idx]
        num_filters = layer.filters
        filter_shape = layer.kernel_size
        strides = layer.strides
        padding = layer.padding.upper()
        W = layer.get_weights()[0]
        num_channels = W.shape[-2]  

        input_shape = self.trained_net.layers[0].input_shape[1:]

        wX, qX = super()._get_layer_data(layer_idx)
        super()._log("\tBuilding patch tensors...")

        tic = time()
        wX_patch_tensor, qX_patch_tensor = self._get_patch_tensors(filter_shape, strides, padding, wX, qX)
        print(f"\tdone. {time() - tic:.2f} seconds.")

        rad = self.alphabet_scalar * median(abs(W.flatten()))
        Q = zeros(W.shape)

        for filter_idx in range(num_filters):
            super()._log(f"\tQuantizing filter {filter_idx} of {num_filters}...")
            tic = time()
            quantized_chan_filter_list = self._quantize_filter2D(
                layer_idx, filter_idx, wX_patch_tensor, qX_patch_tensor, rad
            )
            # Now we need to stack all the channel information together again.
            quantized_filter = zeros((filter_shape[0], filter_shape[1], num_channels))
            for channel_filter in quantized_chan_filter_list:
                channel_idx = channel_filter.channel_idx
                quantized_filter[:, :, channel_idx] = channel_filter.q_filtr

            Q[:, :, :, filter_idx] = quantized_filter

            super()._log(f"\tdone. {time() - tic:.2f} seconds.")

        super()._update_weights(layer_idx, Q)

    def quantize_network(self):

        for layer_idx, layer in enumerate(self.trained_net.layers):
            if layer.__class__.__name__ == "Dense":
                # Use parent class quantize layer
                super()._log(f"Quantizing (dense) layer {layer_idx}...")
                tic = time()
                self._quantize_dense_layer(layer_idx)
                super()._log(f"done. {time() - tic:.2f} seconds.")
            if layer.__class__.__name__ == "Conv2D":
                super()._log(f"Quantizing (Conv2D) layer {layer_idx}...")
                tic = time()
                self._quantize_conv2D_layer(layer_idx)
                super()._log(f"done. {time() - tic:.2f} seconds.")
