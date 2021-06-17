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
from glob import glob

# Define namedtuples for more interpretable return types.
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
        Filename for hdf5 file with datasets wX, qX, transposed.
    alphabet : array
        Scalar quantization alphabet

    Returns
    -------
    q: array
        Quantized neuron.
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
        kernel_size: tuple, strides: tuple, padding: str, rate: tuple, channel_wX: array, channel_qX: array
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
        rate: tuple
            Tuple of integers encoding the dilation rate, if applicable.
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

        kernel_sizes_list = [1, *kernel_size, 1]
        strides_list = [1, *strides, 1]
        if rate:
            rates = [1, *rate, 1]
        else:
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

        #TODO: below you're reshaping a tensor with numpy function!!
        wX_seg = reshape(wX_seg, new_shape)
        qX_seg = reshape(qX_seg, new_shape)



        return SegmentedData(wX_seg=wX_seg, qX_seg=qX_seg)

def _quantize_filter2D_parallel_jit(
        chan_filter: array, channel_idx: int, channel_hf_filename: dict, alphabet: array
    ) -> array:
        """Quantizes a given channel filter as though it were a neuron in a perceptron.

        Parameters
        -----------
        chan_filter: 2D array
            Channel filter to be quantized. 
        channel_idx: int
            Index of channel that we're quantizing. We need this because it appears in the
            name of the dataset in channel_hf_filename.
        channel_hf_filename: dict
            File name that references hdf5 files which contains the patch array for this channel filter.
            This patch array should be a 2D array where the directions of the random
            walk, or feature data, are *rows*.
        alphabet: array
            Quantization alphabet.

        Returns
        -------
        quantized_chan_filters: dict
            Returns a dictionary with (key, value) = (channel_idx, quantized channel filter).
        """

        # Initialize the state variable of the dynamical system,
        # and vectorize the channel filter.
        with h5py.File(f"./{channel_hf_filename}", 'r') as hf:
            u = zeros(hf[f"wX_channel{channel_idx}"].shape[1])
        filter_shape = chan_filter.shape
        chan_filter = reshape(chan_filter, chan_filter.size)
        q_filter = zeros(chan_filter.size)

        # Run the dynamical system on this vectorized channel filter.
        for t in range(chan_filter.size):
            with h5py.File(f"./{channel_hf_filename}", 'r') as hf:
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

class MNISTSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        """Constructs a child class of the Keras Sequence class to generate batches
        of images for ImageNet. 

        Parameters
        -----------
        x_set : 1D-array
            Array of images
        y_set : 1D-array
            Labels for the images.
        batch_size: int
            Specifies how many images to generate in a batch.
        """

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return array(batch_x), array(batch_y)

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        """Constructs a child class of the Keras Sequence class to generate batches
        of images for ImageNet. 

        Parameters
        -----------
        x_set : 1D-array
            Array of images
        y_set : 1D-array
            Labels for the images.
        batch_size: int
            Specifies how many images to generate in a batch.
        """

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return array(batch_x), array(batch_y)

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

    def _get_layer_data_generator(self, layer_idx: int, transpose=False):
        """Gets the input data for the layer at a given index.

        Parameters
        -----------
        layer_idx : int
            Index of the layer.
        transpose: bool
            Whether to transpose the hidden activations or not. This is convenient for quantizing
            Dense layers when reading the hidden feature datas as rows will speed up i/o from hdf5 file.

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


        layer = self.trained_net.layers[layer_idx]
        layer_data_shape = layer.input_shape[1:] if layer.input_shape[0] is None else layer.input_shape
        num_inbound_layers = len(inbound_analog_layers) if layer_idx != 0 else 1
        input_analog_layer = self.trained_net.layers[0]
        # Prebuild the partial models, since they require retracing so you don't want to do them inside a loop.
        if layer_idx > 0:
            prev_trained_model = Model(inputs=input_analog_layer.input,
                                     outputs=[analog_layer.output for analog_layer in inbound_analog_layers])
            prev_quant_model = Model(inputs=self.quantized_net.layers[0].input,
                     outputs=[quant_layer.output for quant_layer in inbound_quant_layers])

        # Preallocate space for h5py file.
        # NOTE:  # Technically num_images is an upper bound, since not every batch has full batch size.
        # I'm implicitly assuming that h5py zerofills pre-allocated space.
        num_images = self.get_data.__len__()*self.get_data.batch_size
        hf_dataset_shape = (num_inbound_layers*num_images, *layer_data_shape)
        if transpose:
            hf_dataset_shape = hf_dataset_shape[::-1]
        hf_filename = f"layer{layer_idx}_data.h5"

        with h5py.File(hf_filename, 'w') as hf:
            for batch_idx in range(self.get_data.__len__()):
                # Grab the batch of data, ignoring labels.
                mini_batch = self.get_data.__getitem__(batch_idx)[0]

                if layer_idx == 0:
                    # No hidden layers to pass data through.
                    wX = mini_batch
                    qX = mini_batch
                else:
                    wX = prev_trained_model.predict_on_batch(mini_batch)
                    qX = prev_quant_model.predict_on_batch(mini_batch)

                if batch_idx == 0:
                    hf.create_dataset("wX", shape=hf_dataset_shape)
                    hf.create_dataset("qX", shape=hf_dataset_shape)

                if transpose:
                    hf["wX"][..., batch_idx*wX.shape[0]:(batch_idx+1)*wX.shape[0]] = wX.T
                    hf["qX"][..., batch_idx*qX.shape[0]:(batch_idx+1)*qX.shape[0]] = qX.T
                else:
                    hf["wX"][batch_idx*wX.shape[0]:(batch_idx+1)*wX.shape[0]] = wX
                    hf["qX"][batch_idx*qX.shape[0]:(batch_idx+1)*qX.shape[0]] = qX


                # Dereference and call garbage collection--just to be safe--to free up memory.
                del mini_batch, wX, qX
                gc.collect()

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
        hf_filename = self._get_layer_data_generator(layer_idx, transpose=True)
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
                    raise exc

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
        patch_mini_batch_size=5000,
        is_quantize_conv2d=True,
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
        is_quantize_conv2d: boolean
-           A boolean indicating whether Conv2D layers should be quantized. Defaults to True.
        """

        self.get_data = get_data
        self.trained_net = network
        self.quantized_net = clone_model(network)
        self.quantized_net.set_weights(network.get_weights())

        self.patch_mini_batch_size = patch_mini_batch_size
        self.is_quantize_conv2d = is_quantize_conv2d

        self.alphabet_scalar = alphabet_scalar
        self.bits = bits
        self.alphabet = linspace(-1, 1, num=int(round(2 ** (bits))))
        self.logger = logger

    def _quantize_channel_parallel_jit(
        self,
        channel_idx: int,
        channel_filters: array,
        hidden_activations_hf_filename: str,
        strides: tuple, 
        padding: str, 
        rate: tuple,
        alphabet: array,
        patch_mini_batch_size=5000,
    ) -> array:
        """For every multi-channel filter, this function quantizes the channel filter at index channel_idx.

        Parameters
        -----------
        channel_idx: int
            Index of the channel to quantize. We only use this because it appears in the naming convention
            of the datasets in the corresponding hdf5 file for this channel's feature data.
        channel_filters: 3D array, of shape (kernel_shape[0], kernel_shape[1], num_filters).
            This is the 3D array that corresponds to slicing a convolutional layer's weight matrix
            along the channel axis.
        hidden_activations_hf_filename: str
            Filename for the hdf5 file which contains the hidden activations from the output of the
            previous layer.
        alphabet : array
            Quantization alphabet.
        patch_mini_batch_size: int
            How many images to load in at a time for the purposes of computing the patch array.
        Returns
        -------
        q_filter: array
            Quantized channel filter, of the same shape as chan_filter.
        """

        filter_shape = channel_filters.shape[0:2]

        # Build the channel patch array.
        super()._log(f"\t\tBuilding patch array for channel {channel_idx}...")
        tic = time()
        channel_hf_filename = self._build_patch_array(channel_idx, filter_shape, 
                strides,
                padding,
                rate,
                hidden_activations_hf_filename, # Reference to hdf5 file that contains wX, qX 
                patch_mini_batch_size,)
        super()._log(f"\t\tdone. {time()-tic:.2f} seconds.")

        num_filters = channel_filters.shape[-1]

        Q_channel = zeros(channel_filters.shape)

        # Now multiprocess quantizing channel filters across filter indices.
        super()._log(f"\t\tQuantizing channel filters (in parallel)...")
        tic = time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_filter = {executor.submit(_quantize_filter2D_parallel_jit,
                channel_filters[:,:,filter_idx],
                channel_idx,
                channel_hf_filename, 
                alphabet): filter_idx for filter_idx in range(num_filters)
            }
            for future in concurrent.futures.as_completed(future_to_filter):
                filter_idx = future_to_filter[future]
                try:
                    # Populate the weights in the slice of Q.
                    quantized_filter = future.result()
                    Q_channel[:, :, filter_idx] = quantized_filter
                except Exception as exc:
                    self._log(f'\t\t\tChannel {channel_idx} Filter {filter_idx} generated an exception: {exc}')
                    raise Exception

        super()._log(f"\t\tdone. {time()-tic:.2f} seconds.")
        # Now delete the hdf5 files that stored the patch arrays for quantizing this layer.
        os.remove(f"./{channel_hf_filename}")

        return Q_channel

    def _build_patch_array(self, channel_idx: int, kernel_size: tuple, strides: tuple, padding: str, rate: tuple, hf_filename: str, mini_batch_size: int) -> tuple:
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
        rate: tuple
            Tuple of integers encoding the dilation rate, if applicable.
        hf_filename: str
            File name for the hdf5 file that contains the wX, qX datasets (transposed). These are the hidden layer
            data used to learn the quantizations at the current layer.
        mini_batch_size: int
            How many examples in a minibatch to load from hf_filename.

        Returns
        -------

        """

        patch_hf_filename = f"channel{channel_idx}_patch_array.h5"
        # TODO: open the hf files AT THE LAST POSSIBLE MINUTE. Close each iteration. This prevents
        # a potentially substantial memory leak.
        num_examples_processed = 0
        with h5py.File(f"./{hf_filename}", 'r') as feature_data_hf:
            total_examples = feature_data_hf["wX"].shape[0]
            with h5py.File(f"./{patch_hf_filename}", 'w') as patch_hf:


                while num_examples_processed < total_examples:

                    actual_mini_batch_size = min(mini_batch_size, total_examples - num_examples_processed)

                    channel_wX = feature_data_hf["wX"][num_examples_processed:num_examples_processed+actual_mini_batch_size, ..., channel_idx]
                    channel_qX = feature_data_hf["qX"][num_examples_processed:num_examples_processed+actual_mini_batch_size, ..., channel_idx]

                    # We have to reshape into a 4 tensor because Tensorflow is picky.
                    # TODO: Maybe batch it instead of feeding in a giant batch?
                    channel_wX = reshape(channel_wX, (*channel_wX.shape, 1))
                    channel_qX = reshape(channel_qX, (*channel_qX.shape, 1))

                    seg_data = _segment_data2D(
                        kernel_size, strides, padding, rate, channel_wX, channel_qX
                    )

                    # Store the directions in our random walk as ROWS because it makes accessing
                    # them substantially faster.
                    if num_examples_processed == 0:
                        #TODO: This estimated shape is good! Preallocate away baby!
                        patches_per_image = seg_data.wX_seg.shape[0]//actual_mini_batch_size
                        estimated_shape = (seg_data.wX_seg.shape[1], patches_per_image*total_examples)
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
                    patch_tensor_shape = patch_hf[f"wX_channel{channel_idx}"].shape
                    # self._log(f"\t\t\t{num_examples_processed} of {total_examples} processed for channel {channel_idx}.")
                    # self._log(f"\t\t\t\tPatch tensors for channel {channel_idx} have shape {patch_tensor_shape}")

                    # Dereference and call garbage collection just to be safe.
                    del seg_data, channel_wX, channel_qX
                    gc.collect()
        return patch_hf_filename

    def _quantize_dense_layer(self, layer_idx: int):

        super()._quantize_layer_parallel(layer_idx)

    def _quantize_conv2D_layer_parallel_jit(self, layer_idx: int):
        # Grab the filename for the hdf5 file which stores the output of the previous
        # hidden layers.
        super()._log("\tFeeding input data through hidden layers...")
        tic = time()
        hf_filename = super()._get_layer_data_generator(layer_idx)
        super()._log(f"\tdone. {time()-tic:.2f} seconds.")

        layer = self.trained_net.layers[layer_idx]
        try:
            # Grab the dilation rate if it exists for this layer. It's relevant for
            # forming the patch arrays.
            rate = layer.dilation_rate
        except:
            rate = None
        W = layer.get_weights()[0]
        rad = self.alphabet_scalar * median(abs(W.flatten()))
        alphabet = rad*self.alphabet
        Q = zeros(W.shape)
        num_channels = W.shape[-2]
        filter_shape = W[0:2]

        if filter_shape == (1,1):
            # Use MSQ, since that's what the data will learn anyways.
            super()._log(f"\t\tFilter shape is (1,1), using MSQ...")
            Q = np.array([_bit_round_parallel(w, layer_alphabet) for w in W.flatten()]).reshape(
                W.shape
            )
        else:
            for channel_idx in range(num_channels):

                # super()._log(f"\tQuantizing filters along channel {channel_idx}...")
                # NOTE: Multiprocessing is only advantageous if there are multiple
                # filters per channel. This need not be the case, e.g. in DepthwiseConv2D
                # layers where the depth_multiplier=1.
                tic = time()
                Q[:, :, channel_idx, :] = self._quantize_channel_parallel_jit(
                                                channel_idx,
                                                W[:, :, channel_idx, :],
                                                hf_filename,
                                                strides=layer.strides,
                                                padding=layer.padding.upper(),
                                                rate=rate,
                                                alphabet=alphabet,
                                                patch_mini_batch_size=self.patch_mini_batch_size,
                                            )
                # super()._log(f"\tdone. {time()-tic:.2f} seconds.")

        # Update the weights of the quantized network at this layer.
        super()._update_weights(layer_idx, Q)

        # Delete the hdf5 file that stores the hidden activations wX, qX datasets.
        os.remove(f"./{hf_filename}")

    def quantize_network(self):

        num_layers = len(self.trained_net.layers)
        for layer_idx, layer in enumerate(self.trained_net.layers):
            if layer.__class__.__name__ == "Dense":
                # Use parent class quantize layer
                super()._log(f"Quantizing (Dense) layer {layer_idx} of {num_layers}...")
                tic = time()
                self._quantize_dense_layer(layer_idx)
                super()._log(f"done. {time() - tic:.2f} seconds.")
            if layer.__class__.__name__ in {"Conv2D", "DepthwiseConv2D"} and self.is_quantize_conv2d:
                super()._log(f"Quantizing ({layer.__class__.__name__}) layer {layer_idx} of {num_layers}...")
                tic = time()
                self._quantize_conv2D_layer_parallel_jit(layer_idx)
                super()._log(f"done. {time() - tic:.2f} seconds.")

