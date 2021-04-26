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
    prod,
    zeros_like,
    load,
)
from math import ceil, floor
from scipy.linalg import norm
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import function as Kfunction
from tensorflow.keras.models import Model, clone_model
from tensorflow.image import extract_patches
from typing import List, Generator
from collections import namedtuple
from itertools import product
from time import time
import concurrent.futures
import h5py
import os
import gc

# Define namedtuples for more interpretable return types.
QuantizedNeuron = namedtuple("QuantizedNeuron", ["layer_idx", "neuron_idx", "q"])
QuantizedFilter = namedtuple(
    "QuantizedFilter", ["layer_idx", "filter_idx", "channel_idx", "q_filtr"]
)
SegmentedData = namedtuple("SegmentedData", ["wX_seg", "qX_seg"])

# Define static functions to use for multiprocessing
def _bit_round_parallel(t: float, alphabet: array) -> float:
    """Rounds a quantity to the nearest atom in the (scaled) quantization alphabet.

    Parameters
    -----------
    t : float
        The value to quantize.
    alphabet : array
        Scalar quantization alphabet.

    Returns
    -------
    bit : float
        The quantized value.
    """

    # Scale the alphabet appropriately.
    return alphabet[argmin(abs(alphabet - t))]

def _quantize_weight_parallel(
    w: float, u: array, X: array, X_tilde: array, alphabet: array
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
    alphabet : array
        Scalar quantization alphabet.

    Returns
    -------
    bit : float
        The quantized value.
    """

    if norm(X_tilde, 2) < 10 ** (-16):
        return 0

    if abs(dot(X_tilde, u)) < 10 ** (-10):
        return _bit_round_parallel(w, alphabet)

    return _bit_round_parallel(dot(X_tilde, u + w * X) / (norm(X_tilde, 2) ** 2), alphabet)

def _quantize_neuron_parallel(
    w: array,
    hf_filename: str,
    alphabet: array,
) -> array:
    """Quantizes a single neuron in a Dense layer.

    Parameters
    -----------
    w: array
        The neuron to be quantized.
    hf_filename: str
        Filename for hdf5 file with datasets wX, qX.
    alphabet : array
        Scalar quantization alphabet

    Returns
    -------
    QuantizedNeuron: NamedTuple
        A tuple with the layer and neuron index, as well as the quantized neuron.
    """
    with h5py.File(hf_filename, 'r') as hf:
        N_ell = hf['wX'].shape[0]
        u = zeros(hf['wX'].shape[1])
        q = zeros(N_ell)
        for t in range(N_ell):
            q[t] = _quantize_weight_parallel(w[t], u, hf['wX'][t, :], hf['qX'][t, :], alphabet)
            u += w[t] * hf['wX'][t, :] - q[t] * hf['qX'][t, :]

    return q

def _segment_data2D(
        kernel_size: tuple, strides: tuple, padding: str, channel_wX: array, channel_qX: array
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
        channel_wX : Array of 2D arrays
            Layer (channel) input for the analog convolutional neural network.
        channel_qX : Array of 2D arrays
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
            images=channel_wX,
            sizes=kernel_sizes_list,
            strides=strides_list,
            rates=rates,
            padding=padding,
        )

        qX_seg = extract_patches(
            images=channel_qX,
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

def _build_patch_array_parallel(channel_idx: int, kernel_size: tuple, strides: tuple, padding: str, hf_filename: str, patch_hf_filename: str, mini_batch_size: int) -> tuple:
        """Returns a 3D tensor whose first two axes encode the 2D tensor of vectorized
        patches of images, and the last axis encodes the channel.

        Parameters
        -----------
        kernel_size: tuple
            Tuple of integers encoding the dimensions of the filter/kernel
        channel_idx: int
            Index to reference the approproiate channel of the feature data.
        strides: tuple
            Tuple of integers encoding the stride information of the filter/kernel
        padding: string
            Padding argument for Conv2D layer.
        hf_filename: str
            File name for the hdf5 file that contains the wX, qX datasets (transposed). These are the hidden layer
            data used to learn the quantizations at the current layer.
        patch_hf_filename: str
            File name of the h5py file to write the image patch array to.
        mini_batch_size: int
            How many examples in a minibatch to load from hf_filename.

        Returns
        -------

        """

        num_examples_processed = 0
        with h5py.File(f"./{hf_filename}", 'r') as feature_data_hf:
            # Remember we transposed the hidden layer data, so number of training examples
            # is the last dimension.
            total_examples = feature_data_hf["wX"].shape[-1]
            with h5py.File(f"./{patch_hf_filename}", 'w') as patch_hf:


                while num_examples_processed < total_examples:

                    actual_mini_batch_size = min(mini_batch_size, total_examples - num_examples_processed)

                    # Remember that we transposed the wX, qX data when we store them in hdf5 files. This is why 
                    # it's channel first here, despite tensorflow adopting the paradigm of being channel last. 
                    # Hence why we also transpose after slicing.
                    channel_wX = feature_data_hf["wX"][channel_idx, ..., num_examples_processed:num_examples_processed+actual_mini_batch_size].T
                    channel_qX = feature_data_hf["qX"][channel_idx, ..., num_examples_processed:num_examples_processed+actual_mini_batch_size].T

                    # We have to reshape into a 4 tensor because Tensorflow is picky.
                    channel_wX = reshape(channel_wX, (*channel_wX.shape, 1))
                    channel_qX = reshape(channel_qX, (*channel_qX.shape, 1))

                    # TODO: Something breaks here if you multiprocess _build_patch_array_parallel() across channels.
                    seg_data = _segment_data2D(
                        kernel_size, strides, padding, channel_wX, channel_qX
                    )

                    # Store the directions in our random walk as ROWS because it makes accessing
                    # them substantially faster.
                    if num_examples_processed == 0:
                        # estimated_shape = (seg_data.wX_seg.shape[1], seg_data.wX_seg.shape[0] * floor(total_examples/mini_batch_size) + total_examples%mini_batch_size)
                        # patch_hf.create_dataset(f"wX_channel{channel_idx}", shape=estimated_shape)
                        # patch_hf.create_dataset(f"qX_channel{channel_idx}", shape=estimated_shape)
                        patch_hf.create_dataset(f"wX_channel{channel_idx}", data = seg_data.wX_seg.T, chunks=True, maxshape=(None,None))
                        patch_hf.create_dataset(f"qX_channel{channel_idx}", data = seg_data.qX_seg.T, chunks=True, maxshape=(None,None))
                    else:
                        # Reshape the h5py datasets and append this mini-batch of patch tensors.
                        patch_hf[f"wX_channel{channel_idx}"].resize((patch_hf[f"wX_channel{channel_idx}"].shape[1]+seg_data.wX_seg.shape[0]), axis = 1)
                        patch_hf[f"wX_channel{channel_idx}"][..., -seg_data.wX_seg.shape[0]:] = seg_data.wX_seg.T

                        patch_hf[f"qX_channel{channel_idx}"].resize((patch_hf[f"qX_channel{channel_idx}"].shape[1]+seg_data.qX_seg.shape[0]), axis = 1)
                        patch_hf[f"qX_channel{channel_idx}"][..., -seg_data.qX_seg.shape[0]:] = seg_data.qX_seg.T

                    # patch_hf[f"wX_channel{channel_idx}"][..., num_examples_processed:num_examples_processed+seg_data.wX_seg.shape[0]] = seg_data.wX_seg.T
                    # patch_hf[f"qX_channel{channel_idx}"][..., num_examples_processed:num_examples_processed+seg_data.wX_seg.shape[0]] = seg_data.qX_seg.T
                    num_examples_processed += actual_mini_batch_size

                    # Dereference and call garbage collection just to be safe.
                    del seg_data, channel_wX, channel_qX
                    gc.collect()

                    

def _quantize_filter2D_parallel(
        img_filter: array, channel_hf_filenames: dict, alphabet: array
    ) -> List[QuantizedFilter]:
        """Quantizes a given filter, or kernel, in a Conv2D layer by quantizing each channel filter
        as though it were a neuron in a perceptron.

        Parameters
        -----------
        img_filter: 3D array
            Filter to be quantized. Note that this filter potentially consists of many channel filters,
            which we quantize independently of each other.
        channel_hf_filenames: dict
            A dictionary with (key, value) = (channel_idx, file name for channel patch array hdf5 file).
            These file names reference hdf5 files which contains the patch arrays for a given channel. 
            Every dataset in this file should be a 2D array where the directions of the random
            walk, or feature data, are *rows*.
        alphabet: array
            Quantization alphabet.

        Returns
        -------
        quantized_chan_filters: dict
            Returns a dictionary with (key, value) = (channel_idx, quantized channel filter).
        """

        is_conv2d_filter = True if len(img_filter.shape) == 3 else False

        # The number of channels is the last dimension by tensorflow convention if the layer is Conv2D.
        num_channels = img_filter.shape[-1] if is_conv2d_filter else 1

        # Initialize a dictionary with (key, value) = (channel_idx, quantized channel filter).
        quantized_chan_filters = {}

        # We're not gonna multiprocess here since we're already multiprocessing this function.
        for channel_idx in range(num_channels):
            if is_conv2d_filter:
                quantized_chan_filters[channel_idx] = _quantize_channel_parallel(
                                                    channel_idx,
                                                    img_filter[:,:,channel_idx],
                                                    channel_hf_filenames[channel_idx],
                                                    alphabet,)
            else:
                quantized_chan_filters[channel_idx] = _quantize_channel_parallel(
                                                    channel_idx,
                                                    img_filter,
                                                    channel_hf_filenames[channel_idx],
                                                    alphabet,)


        if is_conv2d_filter:
            # Now we need to stack all the channel information together into one 3D array.
            quantized_filter = zeros_like(img_filter)
            for channel_idx, quantized_chan_filter in quantized_chan_filters.items():
                quantized_filter[:, :, channel_idx] = quantized_chan_filter
        else:
            quantized_filter = quantized_chan_filters[0]

        return quantized_filter

def _quantize_channel_parallel(
        channel_idx: int,
        chan_filter: array,
        channel_hf_filename: str,
        alphabet: array,
    ) -> QuantizedFilter:
        """Quantizes a single channel filter in a Conv2D layer.

        Parameters
        -----------
        channel_idx: int
            Index of the channel to quantize. We only use this because it appears in the naming convention
            of the datasets in the corresponding hdf5 file for this channel's feature data.
        chan_filter: array
            The channel filter to quantize.
        channel_hf_filename: str
            The filename of the hdf5 file that contains the patch array for this channel. Every dataset
            in this file should be a 2D array where the directions of the random
            walk, or feature data, are *rows*.
        alphabet : array
            Quantization alphabet.

        Returns
        -------
        q_filter: array
            Quantized channel filter, of the same shape as chan_filter.
        """
        with h5py.File(f"./{channel_hf_filename}", 'r') as hf:

            # Initialize the state variable of the dynamical system,
            # and vectorize the channel filter.
            u = zeros(hf[f"wX_channel{channel_idx}"].shape[1])
            filter_shape = chan_filter.shape
            chan_filter = reshape(chan_filter, chan_filter.size)
            q_filter = zeros(chan_filter.size)

            # Run the dynamical system on this vectorized channel filter.
            for t in range(chan_filter.size):
                q_filter[t] = _quantize_weight_parallel(
                    chan_filter[t],
                    u,
                    hf[f"wX_channel{channel_idx}"][t,:],
                    hf[f"qX_channel{channel_idx}"][t,:],
                    alphabet,
                )
                u += chan_filter[t] * hf[f"wX_channel{channel_idx}"][t,:] - q_filter[t] * hf[f"qX_channel{channel_idx}"][t,:]

            # Reshape the quantized channel filter into a 2D array of the appropriate shape.
            q_filter = reshape(q_filter, filter_shape)

            return q_filter

class ImageNetSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, preprocess_func):
        """Constructs a child class of the Keras Sequence class to generate batches
        of images for ImageNet. 

        Parameters
        -----------
        x_set : 1D-array
            Array of paths to images.
        y_set : 1D-array
            Labels for the images.
        batch_size: int
            Specifies how many images to generate in a batch.
        preprocess_func: function
            Preprocessing function to call before yielding images. Examples include the
            preprocessing functions that come packaged with the pretrained keras models.
        """

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.preprocess_func = preprocess_func

    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return array([
            self.preprocess_func(load(file_name))
               for file_name in batch_x]), array(batch_y)

class QuantizedNeuralNetwork:
    def __init__(
        self,
        network: Model,
        batch_size: int,
        get_data: Generator[array, None, None],
        mini_batch_size=32,
        logger=None,
        ignore_layers=[],
        bits=log2(3),
        alphabet_scalar=1,
    ):
        """This is a wrapper class for a tensorflow.keras.models.Model class
        which handles quantizing the weights for Dense layers.

        Parameters
        -----------
        network : Model
            The pretrained neural network.
        batch_size : int,
            How many training examples to use for learning the quantized weights in a
            given layer.
        get_data : Generator
            A generator for yielding training examples for learning the quantized weights.
        mini_batch_size: int
            How many training examples to feed through the hidden layers at a time. We can't feed
            in the entire batch all at once if the batch is large since CPUs and GPUs are memory 
            constrained.
        logger : logger
            A logging object to write updates to. If None, updates are written to stdout.
        ignore_layers : list of ints
            A list of layer indices to indicate which layers are *not* to be quantized.
        bits : float
            How many bits to use for the quantization alphabet. There are 2**bits characters
            in the quantization alphabet.
        alphabet_scalar : float
            A scaling parameter used to adjust the radius of the quantization alphabets for
            each layer.
        """

        self.get_data = get_data

        # The pre-trained network.
        self.trained_net = network

        # This copies the network structure but not the weights.
        self.quantized_net = clone_model(network)

        # Set all the weights to be the same a priori.
        self.quantized_net.set_weights(network.get_weights())

        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

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
        hf_filename : str
            Filename of hdf5 file that contains datasets wX, qX.
        """

        # Determine how many inbound layers there are.
        if layer_idx == 0:
            # Don't need to feed data through hidden layers.
            inbound_analog_layers = None
            inbound_quant_layers = None
        else:
            # Determine whether there is more than one input layer. 
            inbound_analog_nodes = self.trained_net.layers[layer_idx].inbound_nodes
            if len(inbound_analog_nodes) > 1:
                self._log(f"Number of inbound analog nodes = {inbound_analog_nodes}...not sure what to do here!")
            else:
                inbound_analog_layers = inbound_analog_nodes[0].inbound_layers

            inbound_quant_nodes = self.quantized_net.layers[layer_idx].inbound_nodes
            if len(inbound_quant_nodes) > 1:
                self._log(f"Number of inbound quantized nodes = {inbound_quant_nodes}...not sure what to do here!")
            else:
                inbound_quant_layers = inbound_quant_nodes[0].inbound_layers

            # Sanity check that the two networks have the same number of inbound layers
            try:
                assert(len(inbound_analog_layers) == len(inbound_quant_layers))
            except TypeError: 
                # inbound_*_layers is a layer object, not a list
                inbound_analog_layers = [inbound_analog_layers]
                inbound_quant_layers = [inbound_quant_layers]

        # Now tell h5py how big your datasets are going to be. Remember that you'll actually
        # be transposing the hidden layer data since it's faster to access the feature data
        # when they're stored as rows.
        layer = self.trained_net.layers[layer_idx]
        layer_data_shape = layer.input_shape[1:] if layer.input_shape[0] is None else layer.input_shape
        num_inbound_layers = len(inbound_analog_layers) if layer_idx != 0 else 1
        input_analog_layer = self.trained_net.layers[0]
        # Prebuild the partial models, since they require retracing so you don't want to do them inside a loop.
        if layer_idx > 0:
            prev_trained_models =  [Model(inputs=input_analog_layer.input,
                                     outputs=analog_layer.output) for analog_layer in inbound_analog_layers]
            prev_quant_models =  [Model(inputs=self.quantized_net.layers[0].input,
                     outputs=quant_layer.output) for quant_layer in inbound_quant_layers]


        hf_dataset_shape = (num_inbound_layers*self.batch_size, *layer_data_shape)[::-1]
        hf_filename = f"layer{layer_idx}_data.h5"
        with h5py.File(hf_filename, 'w') as hf:
            hf.create_dataset("wX", shape=hf_dataset_shape)
            hf.create_dataset("qX", shape=hf_dataset_shape)

            
            try:
                # Don't ask me why, but for some models like the pretrained models the input layer's input shape
                # is a list of a single tuple.
                if len(input_analog_layer.input_shape) >= 1:
                    input_shape = input_analog_layer.input_shape[0][1:] if input_analog_layer.input_shape[0][0] is None else input_analog_layer.input_shape[0]
            except TypeError:
                # It's not a list, so just access the tuple shape directly.
                input_shape = input_analog_layer.input_shape[1:] if input_analog_layer.input_shape[0] is None else input_analog_layer.input_shape


            # TODO: make get_data() a generator of generators. Then just feed the yielded generators to a newly compiled
            # model which gives the output of a hidden layer.

            num_examples_processed = 0

            while num_examples_processed < self.batch_size:

                # Grab a minibatch of data. Make sure not to request more data
                # than would exceed the total number of samples used to train the layer.
                # This is important if self.batch_size % self.mini_batch_size != 0.
                actual_mini_batch_size = min(self.mini_batch_size, self.batch_size - num_examples_processed)
                mini_batch = zeros((actual_mini_batch_size, *input_shape))

                for sample_idx in range(actual_mini_batch_size):
                    try:
                        mini_batch[sample_idx, :] = next(self.get_data)
                    except StopIteration:
                        # No more samples!
                        self._log(f"\tThe quantization training data generator has reached a StopIteration.")
                        break

                wX = zeros((num_inbound_layers*actual_mini_batch_size, *layer_data_shape))
                qX = zeros((num_inbound_layers*actual_mini_batch_size, *layer_data_shape))

                if layer_idx == 0:
                    # No hidden layers to pass data through.
                    wX = mini_batch
                    qX = mini_batch
                else:
                    # Convert to tensor to prevent retracing.
                    mini_batch = convert_to_tensor(mini_batch, dtype=input_analog_layer.dtype)

                    # For every inbound layer, get the output from passing through that inbound layer
                    for inbound_layer_idx in range(num_inbound_layers):
                        # TODO: cast mini_batch to tensor of appropriate datatype. May prevent retracing.
                        # TODO: Use predict_on_batch instead. Supposedly a lot faster. See
                        # https://stackoverflow.com/a/61287324

                        # Retracing may occur when mini_batch isn't the full batch size.
                        wX = prev_trained_models[inbound_layer_idx].predict_on_batch(mini_batch)
                        qX = prev_quant_models[inbound_layer_idx].predict_on_batch(mini_batch)



                # Append to the hdf5 datasets. Remember to transpose because it's faster to read rows!
                try:
                    hf["wX"][..., num_examples_processed:num_examples_processed+wX.shape[0]] = wX.T
                    hf["qX"][..., num_examples_processed:num_examples_processed+qX.shape[0]] = qX.T
                except Exception as exc:
                    breakpoint()
                    raise exc

                # Dereference and call garbage collection--just to be safe--to free up memory.
                del mini_batch, wX, qX
                gc.collect()

                # Increment the number of images processed!
                num_examples_processed += actual_mini_batch_size

        return hf_filename

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

    def _quantize_layer_parallel(self, layer_idx: int):
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

        self._log("\tFeeding input data through hidden layers...")
        tic = time()
        hf_filename = self._get_layer_data(layer_idx)
        self._log(f"\tdone. {time()-tic:2f} seconds.")

        # Set the radius of the alphabet.
        rad = self.alphabet_scalar * median(abs(W.flatten()))
        layer_alphabet = rad*self.alphabet

        self._log("\tQuantizing neurons (in parallel)...")
        tic = time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Build a dictionary with (key, value) = (q, neuron_idx). This will
            # help us map quantized neurons to the correct neuron index as we call
            # _quantize_neuron asynchronously.
            future_to_neuron = {executor.submit(_quantize_neuron_parallel, W[:, neuron_idx], 
                hf_filename,
                layer_alphabet,
                ): neuron_idx for neuron_idx in range(N_ell_plus_1)}
            for future in concurrent.futures.as_completed(future_to_neuron):
                neuron_idx = future_to_neuron[future]
                try:
                    # Populate the appropriate column in the quantized weight matrix
                    # with the quantized neuron
                    Q[:, neuron_idx] = future.result()
                except Exception as exc:
                    self._log(f'\t\tNeuron {neuron_idx} generated an exception: {exc}')

                self._log(f'\t\tNeuron {neuron_idx} of {N_ell_plus_1} quantized successfully.')

            # Set the weights for the quantized network.
            self._update_weights(layer_idx, Q)
        self._log(f"\tdone. {time()-tic:.2f} seconds.")

        # Now delete the hdf5 file.
        os.remove(f"./{hf_filename}")

    def quantize_network(self):
        """Quantizes all Dense layers that are not specified by the list of ignored layers."""

        # This must be done sequentially.
        num_layers = len(self.trained_net.layers)
        for layer_idx, layer in enumerate(self.trained_net.layers):
            if (
                layer.__class__.__name__ == "Dense"
                and layer_idx not in self.ignore_layers
            ):
                # Only quantize dense layers.
                tic = time()
                self._log(f"Quantizing layer {layer_idx} (in parallel) of {num_layers}...")
                self._quantize_layer_parallel(layer_idx)
                self._log(f"Layer {layer_idx} of {num_layers} quantized successfully in {time() - tic:.2f} seconds.")


class QuantizedCNN(QuantizedNeuralNetwork):

    def __init__(
        self,
        network: Model,
        batch_size: int,
        get_data: Generator[array, None, None],
        mini_batch_size=32,
        logger=None,
        bits=log2(3),
        alphabet_scalar=1,
        patch_mini_batch_size=5000
    ):
        """This is a wrapper class for a tensorflow.keras.models.Model class
        which handles quantizing the weights for Dense and Conv2D layers.

        Parameters
        -----------
        network : Model
            The pretrained neural network.
        batch_size : int,
            How many training examples to use for learning the quantized weights in a
            given layer.
        get_data : Generator
            A generator for yielding training examples for learning the quantized weights.
        mini_batch_size: int
            How many training examples to feed through the hidden layers at a time. We can't feed
            in the entire batch all at once if the batch is large since CPUs and GPUs are memory 
            constrained.
        logger : logger
            A logging object to write updates to. If None, updates are written to stdout.
        bits : float
            How many bits to use for the quantization alphabet. There are 2**bits characters
            in the quantization alphabet.
        alphabet_scalar : float
            A scaling parameter used to adjust the radius of the quantization alphabets for
            each layer.
        patch_mini_batch_size: int
            A separate mini_batch_size used for extracting image patches from hidden convolutional layers.
            Things run really slow if this isn't terribly large, e.g. 32. You have to balance this with
            how much memory you have in RAM. Remember: when you feed in 5000 training examples, you amplify
            how large the patch tensors are by multiplying how many patches come from the image.
        """

        self.get_data = get_data
        self.trained_net = network
        self.quantized_net = clone_model(network)
        self.quantized_net.set_weights(network.get_weights())

        # This quantifies how many images are used in a given batch to train a layer. This is subtly different
        # than the batch_size for the perceptron case because the actual data here are *patches* of images.
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.patch_mini_batch_size = patch_mini_batch_size

        self.alphabet_scalar = alphabet_scalar
        self.bits = bits
        self.alphabet = linspace(-1, 1, num=int(round(2 ** (bits))))
        self.logger = logger

    def _quantize_channel(
        self,
        layer_idx: int,
        filter_idx: int,
        channel_idx: int,
        hf,
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
        hf: hdf5 File object
            Contains the patch tensors for all channels. Every dataset
            in this file should be a 2D array where the directions of the random
            walk, or feature data, are *rows*.
        rad : float
            Scaling factor for the quantization alphabet.

        Returns
        -------
        QuantizedFilter: NamedTuple
            A tuple with the layer, filter, and channel index, as well as the quantized channel.
        """

        B = hf[f"wX_channel0"].shape[1]
        u = zeros(B)
        chan_filtr = self.trained_net.layers[layer_idx].get_weights()[0][:, :, :, filter_idx][:, :, channel_idx]
        filter_shape = chan_filtr.shape
        chan_filtr = reshape(chan_filtr, chan_filtr.size)
        q_filtr = zeros(chan_filtr.size)

        for t in range(chan_filtr.size):
            q_filtr[t] = super()._quantize_weight(
                chan_filtr[t],
                u,
                hf[f"wX_channel{channel_idx}"][t,:],
                hf[f"qX_channel{channel_idx}"][t,:],
                rad,
            )
            u += chan_filtr[t] * hf[f"wX_channel{channel_idx}"][t,:] - q_filtr[t] * hf[f"qX_channel{channel_idx}"][t,:]

        q_filtr = reshape(q_filtr, filter_shape)

        return QuantizedFilter(
                    layer_idx=layer_idx,
                    filter_idx=filter_idx,
                    channel_idx=channel_idx,
                    q_filtr=q_filtr,
                )

    def _quantize_filter2D(
        self, layer_idx: int, filter_idx: int, hf, rad: float
    ) -> List[QuantizedFilter]:
        """Quantizes a given filter, or kernel, in a Conv2D layer by quantizing each channel filter
        as though it were a neuron in a perceptron.

        Parameters
        -----------
        layer_idx : int
            Index of the Conv2D layer.
        filter_idx : int
            Index of the filter in the Conv2D layer.
        hf: hdf5 File object
            Contains the patch tensors for all channels. Every dataset
            in this file should be a 2D array where the directions of the random
            walk, or feature data, are *rows*.
        rad: float
            Scaling factor for the quantization alphabet.

        Returns
        -------
        quantized_chan_filter_list: List[QuantizedFilter]
            Returns a list of quantized channel filters.
        """
        num_channels = self.trained_net.layers[layer_idx].input_shape[-1]
        quantized_chan_filter_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_channel = {executor.submit(self._quantize_channel, layer_idx, 
                filter_idx, 
                channel_idx, 
                hf, 
                rad,): channel_idx for channel_idx in range(num_channels)}
            for future in concurrent.futures.as_completed(future_to_channel):
                channel_idx = future_to_channel[future]
                try:
                    quantized_chan_filter_list += [future.result()]
                except Exception as exc:
                    super()._log(f'Channel {channel_idx} generated an exception: {exc}')
                    raise Exception

        return quantized_chan_filter_list

    def _quantize_dense_layer(self, layer_idx: int):

        super()._quantize_layer_parallel(layer_idx)

    def _get_patch_tensors(self, kernel_size: tuple, strides: tuple, padding: str, hf_filename: str, patch_hf) -> tuple:
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
        hf_filename: str
            Filename for the hdf5 file that contains the wX, qX datasets (transposed). These are the hidden layer
            data used to learn the quantizations at the current layer.
        patch_hf: hdf5 File object
            h5py File ojbect, in write mode, to write the image patch tensor to.

        Returns
        -------

        """
        with h5py.File(f"./{hf_filename}", 'r') as hf:
            # Remember, we transposed wX, qX when we wrote it to hdf5 file which is why it's channel first.
            num_channels = hf["wX"].shape[0]

            for channel_idx in range(num_channels):
                # Remember that we transposed the wX, qX data when we store them in hdf5 files. This is why 
                # it's channel first here, despite tensorflow adopting the paradigm of being channel last. 
                # Hence why we also transpose after slicing.
                channel_wX = hf["wX"][channel_idx, :, :, :].T
                channel_qX = hf["qX"][channel_idx, :, :, :].T

                # We have to reshape into a 4 tensor because Tensorflow is picky.
                channel_wX = reshape(channel_wX, (*channel_wX.shape, 1))
                channel_qX = reshape(channel_qX, (*channel_qX.shape, 1))

                # TODO: You may have to do this in batches as well.
                seg_data = self._segment_data2D(
                    kernel_size, strides, padding, channel_wX, channel_qX
                )

                # Store the directions in our random walk as ROWS because it makes accessing
                # them substantially faster.
                patch_hf.create_dataset(f"wX_channel{channel_idx}", data = seg_data.wX_seg.T)
                patch_hf.create_dataset(f"qX_channel{channel_idx}", data = seg_data.qX_seg.T)

                # Delete temporary arrays to prevent memory leak across loop iterations.
                del seg_data, channel_wX, channel_qX
                gc.collect()

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

        super()._log("\tFeeding input data through hidden layers...")
        tic = time()
        hf_filename = super()._get_layer_data(layer_idx)
        super()._log(f"\tdone. {time()-tic:.2f} seconds.")
        super()._log("\tBuilding patch tensors...")

        # Create hdf5 file to store patch tensors.
        with h5py.File(f"layer{layer_idx}_patch_tensors.h5", 'w') as patch_hf:
            tic = time()
            # This saves the patch tensors to disk, and closes the file object.
            self._get_patch_tensors(filter_shape, strides, padding, hf_filename, patch_hf)
            super()._log(f"\tdone. {time() - tic:.2f} seconds.")

        # Now that the patch tensor is built, we can delete the hdf5 file that stores the 
        # wX, qX datasets.
        os.remove(f"./{hf_filename}")

        rad = self.alphabet_scalar * median(abs(W.flatten()))
        Q = zeros(W.shape)

        # Open hdf5 file object for reading.
        with h5py.File(f"layer{layer_idx}_patch_tensors.h5", 'r') as hf:

            # TODO: multiprocess here. You may need to pass hf as a string of the filename
            # rather than the file object.
            for filter_idx in range(num_filters):
                super()._log(f"\tQuantizing filter {filter_idx} of {num_filters}...")
                tic = time()
                quantized_chan_filter_list = self._quantize_filter2D(
                    layer_idx, filter_idx, hf, rad
                )
                # Now we need to stack all the channel information together again.
                quantized_filter = zeros((filter_shape[0], filter_shape[1], num_channels))
                for channel_filter in quantized_chan_filter_list:
                    channel_idx = channel_filter.channel_idx
                    quantized_filter[:, :, channel_idx] = channel_filter.q_filtr

                Q[:, :, :, filter_idx] = quantized_filter

                super()._log(f"\tdone. {time() - tic:.2f} seconds.")

        super()._update_weights(layer_idx, Q)

        # Now delete the hdf5 file.
        os.remove(f"./layer{layer_idx}_patch_tensors.h5")

    def _quantize_conv2D_layer_parallel(self, layer_idx: int):
        # wX formatted as an array of images. No flattening.
        layer = self.trained_net.layers[layer_idx]
        num_filters = layer.filters
        filter_shape = layer.kernel_size
        strides = layer.strides
        padding = layer.padding.upper()
        W = layer.get_weights()[0]
        num_channels = W.shape[-2]

        input_shape = self.trained_net.layers[0].input_shape[1:]

        # Grab the filename for the hdf5 file which stores the output of the previous
        # hidden layers.
        super()._log("\tFeeding input data through hidden layers...")
        tic = time()
        hf_filename = super()._get_layer_data(layer_idx)
        super()._log(f"\tdone. {time()-tic:.2f} seconds.")

        # Build a dictionary with (key, value) = (channel_idx, file name where we store the vectorized
        # channel data)
        channel_hf_filenames = {channel_idx: f"layer{layer_idx}_channel{channel_idx}_patch_array.h5" 
                                    for channel_idx in range(num_channels)}

        # Now build the hdf5 files which store the ``vectorized'' image patches for each channel.
        # It's convenient to restructure the data this way because it reduces quantizing channel filters
        # in a convolutional layer to the same dynamical system we use to quantize hidden units
        # in a perceptron.

        # NOTE: I tried multiprocessing _build_patch_array_parallel. Strangely enough, it actually runs slower
        # than when you segment the data with a for loop. I have no idea what tensorflow
        # is doing under the hood with extract_patches that causes the slowdown.

        super()._log(f"\tBuilding patch tensors...")
        tic = time()
        for channel_idx in range(num_channels):
            _build_patch_array_parallel(channel_idx, filter_shape, 
                strides,
                padding,
                hf_filename, # Reference to hdf5 file that contains wX, qX 
                channel_hf_filenames[channel_idx], # Filename for hdf5 file to save this channel's patch array to
                self.patch_mini_batch_size,)
        super()._log(f"\tdone. {time() - tic:.2f} seconds.")

        # Now that the channel patch arrays are built, we can delete the hdf5 file that stores the 
        # wX, qX datasets.
        os.remove(f"./{hf_filename}")

        rad = self.alphabet_scalar * median(abs(W.flatten()))
        alphabet = rad*self.alphabet
        Q = zeros(W.shape)

        # Quantize the filters in parallel.
        super()._log("\tQuantizing filters (in parallel)...")
        tic = time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Build a dictionary with (key, value) = (future, channel_idx). This will
            # help us map channel image patches to the correct neuron index as we call
            # _quantize_neuron asynchronously.
            future_to_filter = {executor.submit(_quantize_filter2D_parallel, W[:, :, :, filter_idx], 
                channel_hf_filenames,
                alphabet,
                ): filter_idx for filter_idx in range(num_filters)}
            for future in concurrent.futures.as_completed(future_to_filter):
                filter_idx = future_to_filter[future]
                try:
                    # Populate the weights in the slice of Q.
                    quantized_filter = future.result()
                    Q[:, :, :, filter_idx] = quantized_filter
                except Exception as exc:
                    self._log(f'\t\tFilter {filter_idx} generated an exception: {exc}')
                    raise Exception

                self._log(f'\t\tFilter {filter_idx} of {num_filters} quantized successfully.')

        # Update the weights of the quantized network at this layer.
        super()._update_weights(layer_idx, Q)

        super()._log(f"\tdone. {time() - tic:.2f} seconds.")

        # Now delete the hdf5 files that stored the patch arrays for quantizing this layer.
        for _, filename in channel_hf_filenames.items():
            os.remove(f"./{filename}")

    def _quantize_depthwise_conv2d_layer_parallel(self, layer_idx: int):
        # wX formatted as an array of images. No flattening.
        layer = self.trained_net.layers[layer_idx]
        depth_multiplier = layer.depth_multiplier
        filter_shape = layer.kernel_size
        strides = layer.strides
        padding = layer.padding.upper()
        W = layer.get_weights()[0]
        num_channels = W.shape[-2]  

        input_shape = self.trained_net.layers[0].input_shape[1:]

        # Grab the filename for the hdf5 file which stores the output of the previous
        # hidden layers.
        super()._log("\tFeeding input data through hidden layers...")
        tic = time()
        hf_filename = super()._get_layer_data(layer_idx)
        super()._log(f"\tdone. {time()-tic:.2f} seconds.")

        # Build a dictionary with (key, value) = (channel_idx, file name where we store the vectorized
        # channel data)
        channel_hf_filenames = {channel_idx: f"layer{layer_idx}_channel{channel_idx}_patch_array.h5" 
                                    for channel_idx in range(num_channels)}

        # Now build the hdf5 files which store the ``vectorized'' image patches for each channel.
        # It's convenient to restructure the data this way because it reduces quantizing channel filters
        # in a convolutional layer to the same dynamical system we use to quantize hidden units
        # in a perceptron.

        super()._log(f"\tBuilding patch tensors...")
        tic = time()
        for channel_idx in range(num_channels):
            _build_patch_array_parallel(channel_idx, filter_shape, 
                strides,
                padding,
                hf_filename, # Reference to hdf5 file that contains wX, qX 
                channel_hf_filenames[channel_idx], # Filename for hdf5 file to save this channel's patch array to
                self.patch_mini_batch_size,)
        super()._log(f"\tdone. {time() - tic:.2f} seconds.")

        # Now that the channel patch arrays are built, we can delete the hdf5 file that stores the 
        # wX, qX datasets.
        os.remove(f"./{hf_filename}")

        rad = self.alphabet_scalar * median(abs(W.flatten()))
        alphabet = rad*self.alphabet
        Q = zeros(W.shape)

        # Quantize the filters in parallel.
        super()._log("\tQuantizing filters (in parallel)...")
        tic = time()
        for depth_idx in range(depth_multiplier):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Build a dictionary with (key, value) = (future, channel_idx). This will
                # help us map channel image patches to the correct neuron index as we call
                # _quantize_neuron asynchronously.
                future_to_filter = {executor.submit(_quantize_filter2D_parallel, W[:, :, filter_idx, depth_idx], 
                    channel_hf_filenames,
                    alphabet,
                    ): filter_idx for filter_idx in range(num_channels)}
                for future in concurrent.futures.as_completed(future_to_filter):
                    filter_idx = future_to_filter[future]
                    try:
                        # Populate the weights in the slice of Q.
                        quantized_filter = future.result()
                        Q[:, :, filter_idx, depth_idx] = quantized_filter
                    except Exception as exc:
                        self._log(f'\t\tDepth {depth_idx} Filter {filter_idx} generated an exception: {exc}')
                        raise Exception

                    self._log(f'\t\tDepth {depth_idx} Filter {filter_idx} of {num_channels} quantized successfully.')

        # Update the weights of the quantized network at this layer.
        super()._update_weights(layer_idx, Q)

        super()._log(f"\tdone. {time() - tic:.2f} seconds.")

        # Now delete the hdf5 files that stored the patch arrays for quantizing this layer.
        for _, filename in channel_hf_filenames.items():
            os.remove(f"./{filename}")

    def quantize_network(self):

        num_layers = len(self.trained_net.layers)
        for layer_idx, layer in enumerate(self.trained_net.layers):
            if layer.__class__.__name__ == "Dense":
                # Use parent class quantize layer
                super()._log(f"Quantizing (Dense) layer {layer_idx} of {num_layers}...")
                tic = time()
                self._quantize_dense_layer(layer_idx)
                super()._log(f"done. {time() - tic:.2f} seconds.")
            if layer.__class__.__name__ == "Conv2D":
                super()._log(f"Quantizing ({layer.__class__.__name__}) layer {layer_idx} of {num_layers}...")
                tic = time()
                # self._quantize_conv2D_layer(layer_idx)
                self._quantize_conv2D_layer_parallel(layer_idx)
                super()._log(f"done. {time() - tic:.2f} seconds.")

            if layer.__class__.__name__ == "DepthwiseConv2D":
                super()._log(f"Quantizing ({layer.__class__.__name__}) layer {layer_idx} of {num_layers}...")
                tic = time()
                # self._quantize_conv2D_layer(layer_idx)
                self._quantize_depthwise_conv2d_layer_parallel(layer_idx)
                super()._log(f"done. {time() - tic:.2f} seconds.")

