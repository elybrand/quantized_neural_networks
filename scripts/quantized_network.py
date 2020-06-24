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
import time

QuantizedNeuron = namedtuple("QuantizedNeuron", ["layer_idx", "neuron_idx", "q", "U"])
QuantizedFilter = namedtuple(
    "QuantizedFilter", ["layer_idx", "filter_idx", "channel_idx", "q_filtr", "U"]
)
SegmentedData = namedtuple("SegmentedData", ["wX_seg", "qX_seg"])
GreedyDirections = namedtuple("GreedyDirection", ["dir_idx", "w", "wX", "qX"])
SortedDirections = namedtuple("SortedDirections", ["wX", "qX", "permutation"])


class QuantizedNeuralNetwork:
    def __init__(
        self,
        network: Model,
        batch_size: int,
        get_data: Generator[array, None, None],
        logger=None,
        ignore_layers=[],
        order=1,
        bits=log2(3),
        alphabet_scalar=1,
    ):
        """
        CAVEAT: Bias terms are not quantized!
        REMEMBER: TensorFlow flips everything for you. Networks act via

        batch_size x N_ell     N_ell x N_{ell+1}

        [ -- X_1^T -- ]     [  |     |       |  ]
        [ -- X_2^T -- ]     [ w_1   w_2     w_3 ]
        [     .       ]     [  |     |       |  ]
        [     .       ]
        [     .       ]
        [ -- X_B^T -- ]

        That means our residual matrix for quantizing the j^th neuron (i.e. w_j as detailed above) will be of the form

            N_ell x batch_size

        [ -- u1^T --      ]
        [     .           ]
        [     .           ]
        [     .           ]
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

        self.alphabet_scalar = alphabet_scalar

        # Create a dictionary encoding which layers are Dense, and what their dimensions are.
        self.layer_dims = {
            layer_idx: layer.get_weights()[0].shape
            for layer_idx, layer in enumerate(network.layers)
            if layer.__class__.__name__ == "Dense"
        }

        self.order = order

        # This determines the alphabet. There will be 2**bits atoms in our alphabet.
        self.bits = bits

        # Construct the (unscaled) alphabet. Layers will scale this alphabet based on the
        # distribution of that layer's weights.
        self.alphabet = linspace(-1, 1, num=int(round(2 ** (bits))))

        self.logger = logger

        self.ignore_layers = ignore_layers

    def bit_round(self, t: float, rad: float) -> int:

        # Scale the alphabet appropriately.
        layer_alphabet = rad * self.alphabet
        return layer_alphabet[argmin(abs(layer_alphabet - t))]

    def quantize_weight(
        self, w: float, u: array, X: array, X_tilde: array, rad: float
    ) -> float:
        # This is undefined if X_tilde is zero. In this case, return 0.

        if norm(X_tilde, 2) < 10 ** (-16):
            return 0

        if abs(dot(X_tilde, u)) < 10 ** (-10):
            return self.bit_round(w, rad)

        return self.bit_round(dot(X_tilde, u + w * X) / (norm(X_tilde, 2) ** 2), rad)

    # One-two step
    def quantize_neuron2(
        self, layer_idx: int, neuron_idx: int, wX: array, qX: array, rad=1
    ) -> QuantizedNeuron:
        u_init = zeros(self.batch_size)
        w = self.trained_net.layers[layer_idx].get_weights()[0][:, neuron_idx]
        N_ell = w.shape[0]
        q = zeros(N_ell)

        # Since we're iterating over two steps at a time, we only need half as many residual vectors
        # to keep track of.
        U = zeros((int(1.0 * N_ell / 2), self.batch_size))

        # State variables for running the two competing strategies of two one-step iterations, and
        # one two-step iteration.
        U1 = zeros((2, self.batch_size))
        U2 = zeros(self.batch_size)

        # Two first order steps.
        q11 = self.quantize_weight(w[0], u_init, wX[:, 0], qX[:, 0], rad)
        U1[0, :] = u_init + w[0] * wX[:, 0] - q11 * qX[:, 0]
        q12 = self.quantize_weight(w[1], U1[0, :], wX[:, 1], qX[:, 1], rad)
        U1[1, :] = U1[0, :] + w[1] * wX[:, 1] - q12 * qX[:, 1]

        # One second order step. You have to brute force search all possible ternary pairs here.
        alphabet = (-rad, 0, rad)
        q21, q22 = (nan, nan)
        for (p1, p2) in product(alphabet, alphabet):
            candidate_U2 = (
                u_init + w[0] * wX[:, 0] - p1 * qX[:, 0] + w[1] * wX[:, 1] - p2 * qX[:, 1]
            )
            if norm(candidate_U2) < norm(U2) or q21 is nan:
                U2 = candidate_U2
                q21, q22 = p1, p2

        # Now choose the pair of bits which minimize the norm of the residual.
        if norm(U1[1, :]) < norm(U2):
            q[0], q[1] = q11, q12
            U[0, :] = U1[1, :]
        else:
            q[0], q[1] = q21, q22
            U[0, :] = U2

        # Now repeat the procedure.
        for t in range(2, N_ell - 1, 2):

            U1 = zeros((2, self.batch_size))
            U2 = zeros(self.batch_size)

            q11 = self.quantize_weight(
                w[t], U[int(t / 2) - 1, :], wX[:, t], qX[:, t], rad
            )
            U1[0, :] = U[int(t / 2) - 1, :] + w[t] * wX[:, t] - q11 * qX[:, t]
            q12 = self.quantize_weight(w[t], U1[0, :], wX[:, t], qX[:, t], rad)
            U1[1, :] = U1[0, :] + w[t] * wX[:, t] - q12 * qX[:, t]

            alphabet = (-rad, 0, rad)
            q21, q22 = (nan, nan)
            for (p1, p2) in product(alphabet, alphabet):
                candidate_U2 = (
                    U[int(t / 2) - 1, :]
                    + w[t] * wX[:, t]
                    - p1 * qX[:, t]
                    + w[t + 1] * wX[:, t + 1]
                    - p2 * qX[:, t + 1]
                )
                if norm(candidate_U2) < norm(U2) or q21 is nan:
                    U2 = candidate_U2
                    q21, q22 = p1, p2

            if norm(U1[1, :]) < norm(U2):
                q[t], q[t + 1] = q11, q12
                U[int(t / 2), :] = U1[1, :]
            else:
                q[t], q[t + 1] = q21, q22
                U[int(t / 2), :] = U2

        qNeuron = QuantizedNeuron(layer_idx=layer_idx, neuron_idx=neuron_idx, q=q, U=U)
        return qNeuron

    def quantize_neuron(
        self,
        layer_idx: int,
        neuron_idx: int,
        wX: array,
        qX: array,
        rad=1,
        permutation=None,
    ) -> QuantizedNeuron:

        N_ell = wX.shape[1]
        u_init = zeros(self.batch_size)
        w = self.trained_net.layers[layer_idx].get_weights()[0][:, neuron_idx]
        if permutation is not None:
            if self.logger:
                self.logger.info("\tPermuting weights...")
            w = [w[idx] for idx in permutation]
        q = zeros(N_ell)
        U = zeros((N_ell, self.batch_size))
        # Take columns of the data matrix, since the samples are given via the rows.
        q[0] = self.quantize_weight(w[0], u_init, wX[:, 0], qX[:, 0], rad)
        U[0, :] = u_init + w[0] * wX[:, 0] - q[0] * qX[:, 0]
        for t in range(1, N_ell):
            q[t] = self.quantize_weight(w[t], U[t - 1, :], wX[:, t], qX[:, t], rad)
            U[t, :] = U[t - 1, :] + w[t] * wX[:, t] - q[t] * qX[:, t]

        return QuantizedNeuron(layer_idx=layer_idx, neuron_idx=neuron_idx, q=q, U=U)

    def sort_directions(self, wX: array, qX: array):
        # Sort directions so that each step maximizes the absolute correlation between the chosen step
        # and the previous step.

        # Normalize all of the columns of wX for comparison.
        wX_unit = array(
            [
                row / norm(row) if norm(row) > 10 ** (-16) else zeros(row.shape)
                for row in wX.T
            ]
        ).T

        # Compute all absolute correlations once and for all so we can quick reference later.
        gramian = abs(wX_unit.T @ wX_unit)
        new_wX = zeros(wX.shape)
        new_qX = zeros(qX.shape)

        # Start with the first one for now.
        new_wX[:, 0] = wX[:, 0]
        new_qX[:, 0] = qX[:, 0]
        # Keep track of the directions you've chosen so far.
        used_idxs = [0]
        for t in range(1, new_wX.shape[1]):
            # Select the relevant row from the gramian.
            # look at absolute dot products.
            inner_prods = gramian[used_idxs[-1], :]
            # Remove those indices which have already been used.
            inner_prods = array(
                [[i, x] for i, x in enumerate(inner_prods) if i not in used_idxs]
            )
            # Find the index of the vector which maximizes the absolute correlation.
            # Add that direction to new_wX and update the list of used_idxs.
            max_inner_prod = max(inner_prods[:, 1])
            next_dir_idx = int([i for (i, x) in inner_prods if x == max_inner_prod][0])

            new_wX[:, t] = wX[:, next_dir_idx]
            new_qX[:, t] = qX[:, next_dir_idx]
            used_idxs += [next_dir_idx]

        return SortedDirections(wX=new_wX, qX=new_qX, permutation=tuple(used_idxs))

    def reorder_bits(self, q: array, permutation: tuple):
        new_q = zeros(q.shape)
        for idx, perm_coord in enumerate(permutation):
            new_q[perm_coord] = q[idx]
        return new_q

    def get_layer_data(self, layer_idx: int):
        W = self.trained_net.layers[layer_idx].get_weights()[0]
        N_ell, N_ell_plus_1 = W.shape
        wX = zeros((self.batch_size, N_ell))
        qX = zeros((self.batch_size, N_ell))
        if layer_idx == 0:
            # Data are assumed to be independent.
            wX = zeros((self.batch_size, N_ell))
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

            input_size = self.trained_net.layers[0].get_weights()[0].shape[0]
            wBatch = zeros((self.batch_size, input_size))
            for sample_idx in range(self.batch_size):
                try:
                    wBatch[sample_idx, :] = next(self.get_data)
                except StopIteration:
                    # No more samples!
                    break
            qBatch = wBatch

            wX = prev_trained_output([wBatch])[0]
            qX = prev_quant_output([qBatch])[0]

        return (wX, qX)

    def update_weights(self, layer_idx: int, Q: array):
        # Update the quantized network. Use the same bias vector as in the analog network for now.
        if self.trained_net.layers[layer_idx].use_bias:
            bias = self.trained_net.layers[layer_idx].get_weights()[1]
            self.quantized_net.layers[layer_idx].set_weights([Q, bias])
        else:
            self.quantized_net.layers[layer_idx].set_weights([Q])

    def quantize_layer(self, layer_idx: int, order: int, use_greedy: bool):
        W = self.trained_net.layers[layer_idx].get_weights()[0]
        N_ell, N_ell_plus_1 = W.shape
        # Placeholder for the weight matrix in the quantized network.
        Q = zeros(W.shape)
        N_ell_plus_1 = W.shape[1]
        wX, qX = self.get_layer_data(layer_idx)

        # Set the radius of the alphabet.
        rad = self.alphabet_scalar * median(abs(W.flatten()))

        # Sort all directions beforehand.
        if use_greedy:
            sorted_dirs = self.sort_directions(wX, qX)

        for neuron_idx in range(N_ell_plus_1):
            if order == 1 and not use_greedy:
                qNeuron = self.quantize_neuron(layer_idx, neuron_idx, wX, qX, rad)
                Q[:, neuron_idx] = qNeuron.q
            if order == 2 and not use_greedy:
                qNeuron = self.quantize_neuron2(layer_idx, neuron_idx, wX, qX, rad)
                Q[:, neuron_idx] = qNeuron.q
            if use_greedy:

                w = self.trained_net.layers[layer_idx].get_weights()[0][:, neuron_idx]
                new_w = [w[idx] for idx in sorted_dirs.permutation]

                qNeuron = self.quantize_neuron(
                    layer_idx,
                    neuron_idx,
                    sorted_dirs.wX,
                    sorted_dirs.qX,
                    rad,
                    permutation=sorted_dirs.permutation,
                )
                # Permute the bit string to what it should be.
                Q[:, neuron_idx] = self.reorder_bits(qNeuron.q, sorted_dirs.permutation)

            if self.logger:
                self.logger.info(
                    f"\tFinished quantizing neuron {neuron_idx} of {N_ell_plus_1}"
                )

            self.update_weights(layer_idx, Q)

    def quantize_neuron_mp(
        self, layer_idx: int, neuron_idx: int, wX: array, qX: array, rad=1
    ) -> QuantizedNeuron:
        d, N = wX.shape
        w = self.trained_net.layers[layer_idx].get_weights()[0][:, neuron_idx]
        q = zeros(N)
        residual = wX @ w
        layer_alphabet = rad * self.alphabet
        permutation = []
        tic = time.perf_counter()
        qX_unit = array(
            [x / norm(x) if norm(x) > 10 ** (-16) else zeros(d) for x in qX.T]
        ).T
        toc = time.perf_counter()
        # self.logger.info(f"\t\t{toc-tic:.2f} seconds to normalize qX")

        # TODO: maybe you can cleverly keep track of indexing and delete as you go?
        # This may speed things up by a constant factor, but not by an order of magnitude.
        for t in range(N):
            # Find the vector whose direction is maximally aligned with the residual.
            tic = time.perf_counter()
            next_dir_idx = argmax(
                [
                    abs(x @ residual) if idx not in permutation else -inf
                    for idx, x in enumerate(qX_unit.T)
                ]
            )
            toc = time.perf_counter()
            # self.logger.info(f"\t\t{toc-tic:.2f} seconds to find next direction")
            x = qX[:, next_dir_idx]
            tic = time.perf_counter()
            q[t] = layer_alphabet[
                argmin(
                    norm(array([residual - bit * x for bit in layer_alphabet]), axis=1)
                )
            ]
            toc = time.perf_counter()
            # self.logger.info(f"\t\t{toc-tic:.2f} seconds to find best bit")
            permutation += [next_dir_idx]
            residual -= q[t] * x

        # Permute q so that it follows the proper ordering of the feature data.
        q = self.reorder_bits(q, permutation)
        return QuantizedNeuron(layer_idx=layer_idx, neuron_idx=neuron_idx, U=None, q=q)

    def quantize_layer_mp(self, layer_idx: int):
        """
        Uses matching pursuit to calculate the bits.
        """

        wX, qX = self.get_layer_data(layer_idx)
        W = self.trained_net.layers[layer_idx].get_weights()[0]
        N = W.shape[1]
        Q = zeros(W.shape)
        # rad = self.alphabet_scalar * median(abs(W.flatten()))
        rad = 1

        for neuron_idx in range(N):
            tic = time.perf_counter()
            qNeuron = self.quantize_neuron_mp(layer_idx, neuron_idx, wX, qX, rad)
            toc = time.perf_counter()
            # self.logger.info(f"\t{toc-tic:.2f} seconds to quantize neuron  {neuron_idx}")
            Q[:, neuron_idx] = qNeuron.q

            # if self.logger:
            #     self.logger.info(f"\tFinished quantizing neuron {neuron_idx} of {N}")

        self.update_weights(layer_idx, Q)

    def quantize_network(self, use_greedy=False):

        # This must be done sequentially.
        for layer_idx, layer in enumerate(self.trained_net.layers):
            if (
                layer.__class__.__name__ == "Dense"
                and layer_idx not in self.ignore_layers
            ):
                # Only quantize dense layers.
                if self.logger:
                    self.logger.info(f"Quantizing layer {layer_idx}...")
                # self.quantize_layer(layer_idx, order=1, use_greedy=False)
                self.quantize_layer_mp(layer_idx)


class QuantizedCNN(QuantizedNeuralNetwork):

    # TODO: Add ignore_layer option

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

        # The pre-trained network.
        self.trained_net = network

        # This copies the network structure but not the weights.
        self.quantized_net = clone_model(network)
        self.quantized_net.set_weights(network.get_weights())

        # This determines the alphabet. There will be 2**bits atoms in our alphabet.
        self.bits = bits

        # Construct the (unscaled) alphabet. Layers will scale this alphabet based on the
        # distribution of that layer's weights.
        self.alphabet = linspace(-1, 1, num=int(round(2 ** (bits))))

        # This quantifies how many images are used in a given batch to train a layer. This is subtly different
        # than the batch_size for the perceptron case because the actual data here are *patches* of images.
        self.batch_size = batch_size

        self.alphabet_scalar = alphabet_scalar

        self.logger = logger

    def segment_data2D(
        self, kernel_size: tuple, strides: tuple, padding, wX: array, qX: array
    ) -> SegmentedData:
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

        # Now at this point, we don't really care about how the patches are grouped. So we reshape them into a 2 tensor, where the patches
        # are vectorized and stored in the rows.
        new_shape = (wX_seg.shape[0] * wX_seg.shape[1] * wX_seg.shape[2], wX_seg.shape[3])
        wX_seg = reshape(wX_seg, new_shape)
        qX_seg = reshape(qX_seg, new_shape)

        return SegmentedData(wX_seg=wX_seg, qX_seg=qX_seg)

    def quantize_filter2D(
        self, layer_idx: int, filter_idx: int, wX: array, qX: array, rad: float
    ) -> List[QuantizedFilter]:

        # Each channel has its own filter, so we need to split by channel. We assume the number of channels
        # is the last dimension in the tensor, since this is how Tensorflow formats it (as compared to Theano, which is first).
        num_channels = wX.shape[-1]
        layer = self.trained_net.layers[layer_idx]
        kernel_size = layer.kernel_size
        strides = layer.strides
        padding = (
            layer.padding.upper()
        )  # Have to capitalize this apparently...why keras and TF don't coordinate this is beyond me.
        quantized_chan_filter_list = []
        for channel_idx in range(num_channels):

            # Segment the channel data into patches.
            channel_wX = wX[:, :, :, channel_idx]
            channel_qX = qX[:, :, :, channel_idx]

            # We have to reshape into a 4 tensor because Tensorflow is picky.
            channel_wX = reshape(channel_wX, (*channel_wX.shape, 1))
            channel_qX = reshape(channel_qX, (*channel_qX.shape, 1))

            # This will return a tensor where the patches have been flattened as rows.
            seg_data = self.segment_data2D(
                kernel_size, strides, padding, channel_wX, channel_qX
            )
            channel_wX_patches = seg_data.wX_seg
            channel_qX_patches = seg_data.qX_seg

            chan_filtr = layer.get_weights()[0][:, :, :, filter_idx][:, :, channel_idx]
            # Flatten the filter.
            chan_filtr = reshape(chan_filtr, chan_filtr.size)
            # Now quantize the filter as if it were a neuron in a perceptron, i.e. a column vector.
            # Here, B represents the patch batch (!) size.
            B = seg_data.wX_seg.shape[0]
            u_init = zeros(B)
            q_filtr = zeros(chan_filtr.size)
            U = zeros((chan_filtr.size, B))
            q_filtr[0] = super().quantize_weight(
                chan_filtr[0],
                u_init,
                channel_wX_patches[:, 0],
                channel_qX_patches[:, 0],
                rad,
            )
            U[0, :] = (
                u_init
                + chan_filtr[0] * channel_wX_patches[:, 0]
                - q_filtr[0] * channel_qX_patches[:, 0]
            )

            for t in range(1, chan_filtr.size):
                q_filtr[t] = super().quantize_weight(
                    chan_filtr[t],
                    U[t - 1, :],
                    channel_wX_patches[:, t],
                    channel_qX_patches[:, t],
                    rad,
                )
                U[t, :] = (
                    U[t - 1, :]
                    + chan_filtr[t] * channel_wX_patches[:, t]
                    - q_filtr[t] * channel_qX_patches[:, t]
                )

            q_filtr = reshape(q_filtr, kernel_size)
            quantized_chan_filter_list += [
                QuantizedFilter(
                    layer_idx=layer_idx,
                    filter_idx=filter_idx,
                    channel_idx=channel_idx,
                    q_filtr=q_filtr,
                    U=U,
                )
            ]

        return quantized_chan_filter_list

    def quantize_dense_layer(self, layer_idx: int):

        input_shape = self.trained_net.layers[0].input_shape[1:]
        W = self.trained_net.layers[layer_idx].get_weights()[0]
        Q = zeros(W.shape)
        N_ell_plus_1 = W.shape[1]
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
            prev_trained_output = Kfunction(
                [self.trained_net.layers[0].input],
                [self.trained_net.layers[layer_idx - 1].output],
            )
            prev_quant_output = Kfunction(
                [self.quantized_net.layers[0].input],
                [self.quantized_net.layers[layer_idx - 1].output],
            )

            # I wonder if this is where you run into memory issues...forcing all the data through the network at once
            # rather than doing it in batches.
            wX = prev_trained_output(batch)[0]
            qX = prev_quant_output(batch)[0]

        # Set the radius of the alphabet.
        rad = self.alphabet_scalar * median(abs(W.flatten()))

        # # Now quantize the neurons.
        # with ThreadPoolExecutor() as executor:
        #     future_to_neuron = {
        #         executor.submit(
        #             self.quantize_neuron, layer_idx, neuron_idx, wX, qX, rad
        #         ): neuron_idx
        #         for neuron_idx in range(N_ell_plus_1)
        #     }
        #     for future in as_completed(future_to_neuron):
        #         neuron_idx = future_to_neuron[future]
        #         try:
        #             qNeuron = future.result()
        #         except Exception as exc:
        #             if self.logger:
        #                 self.logger.error(
        #                     f"Error quantizing neuron {neuron_idx} in layer {layer_idx}: {exc}"
        #                 )
        #             else:
        #                 print(
        #                     f"Error quantizing neuron {neuron_idx} in layer {layer_idx}: {exc}"
        #                 )
        #         # Update quantized weight matrix.
        #         Q[:, neuron_idx] = qNeuron.q
        #         if self.logger:
        #             self.logger.info(
        #                 f"\tFinished quantizing neuron {neuron_idx} of {N_ell_plus_1}"
        #             )

        for neuron_idx in range(N_ell_plus_1):
            qNeuron = self.quantize_neuron(layer_idx, neuron_idx, wX, qX, rad)
            # Update quantized weight matrix.
            Q[:, neuron_idx] = qNeuron.q
            if self.logger:
                self.logger.info(
                    f"\tFinished quantizing neuron {neuron_idx} of {N_ell_plus_1}"
                )
        # Update the quantized network. Use the same bias vector as in the analog network for now.
        if self.trained_net.layers[layer_idx].use_bias:
            bias = self.trained_net.layers[layer_idx].get_weights()[1]
            self.quantized_net.layers[layer_idx].set_weights([Q, bias])
        else:
            self.quantized_net.layers[layer_idx].set_weights([Q])

    def quantize_conv2D_layer(self, layer_idx: int):
        # wX formatted as an array of images. No flattening.
        num_filters = self.trained_net.layers[layer_idx].filters
        filter_shape = self.trained_net.layers[layer_idx].kernel_size
        input_shape = self.trained_net.layers[0].input_shape[1:]
        W = self.trained_net.layers[layer_idx].get_weights()[0]
        num_channels = W.shape[
            -2
        ]  # The last index is the number of filters in this case.
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
            prev_trained_output = Kfunction(
                [self.trained_net.layers[0].input],
                [self.trained_net.layers[layer_idx - 1].output],
            )
            prev_quant_output = Kfunction(
                [self.quantized_net.layers[0].input],
                [self.quantized_net.layers[layer_idx - 1].output],
            )

            wX = prev_trained_output([batch])[0]
            qX = prev_quant_output([batch])[0]

        # Set the radius of the alphabet.
        rad = self.alphabet_scalar * median(abs(W.flatten()))

        Q = zeros(W.shape)

        # # Now quantize the neurons.
        # with ThreadPoolExecutor() as executor:
        #     future_to_filter = {
        #         executor.submit(
        #             self.quantize_filter2D, layer_idx, filter_idx, wX, qX, rad
        #         ): filter_idx
        #         for filter_idx in range(num_filters)
        #     }
        #     for future in as_completed(future_to_filter):
        #         filter_idx = future_to_filter[future]
        #         try:
        #             quantized_chan_filter_list = future.result()
        #         except Exception as exc:
        #             if self.logger:
        #                 self.logger.error(
        #                     f"Error quantizing filter {filter_idx} in layer {layer_idx}: {exc}"
        #                 )
        #             else:
        #                 print(
        #                     f"Error quantizing neuron {filter_idx} in layer {layer_idx}: {exc}"
        #                 )

        #         # Now we need to stack all the channel information together again.
        #         N, B = quantized_chan_filter_list[0].U.shape
        #         filter_U = zeros((N, B, num_channels))
        #         # Again, following Tensorflow convention that the channel information is the l ast component.
        #         quantized_filter = zeros((filter_shape[0], filter_shape[1], num_channels))
        #         for channel_filter in quantized_chan_filter_list:
        #             channel_idx = channel_filter.channel_idx
        #             quantized_filter[:, :, channel_idx] = channel_filter.q_filtr

        #         Q[:, :, :, filter_idx] = quantized_filter

        #         if self.logger:
        #             self.logger.info(
        #                 f"\tFinished quantizing filter {filter_idx} of {num_filters}"
        #             )

        for filter_idx in range(num_filters):
            quantized_chan_filter_list = self.quantize_filter2D(
                layer_idx, filter_idx, wX, qX, rad
            )
            # Now we need to stack all the channel information together again.
            # Again, following Tensorflow convention that the channel information is the last component.
            quantized_filter = zeros((filter_shape[0], filter_shape[1], num_channels))
            for channel_filter in quantized_chan_filter_list:
                channel_idx = channel_filter.channel_idx
                quantized_filter[:, :, channel_idx] = channel_filter.q_filtr

            Q[:, :, :, filter_idx] = quantized_filter

            if self.logger:
                self.logger.info(
                    f"\tFinished quantizing filter {filter_idx} of {num_filters}"
                )

        # Now update the layer in the quantized network. Leave the bias the same for now.
        if self.trained_net.layers[layer_idx].use_bias:
            b = self.quantized_net.layers[layer_idx].get_weights()[1]
            self.quantized_net.layers[layer_idx].set_weights([Q, b])
        else:
            self.quantized_net.layers[layer_idx].set_weights([Q])

    def quantize_network(self):

        for layer_idx, layer in enumerate(self.trained_net.layers):
            if layer.__class__.__name__ == "Dense":
                # Use parent class quantize layer
                if self.logger:
                    self.logger.info(f"Quantizing (dense) layer {layer_idx}...")
                self.quantize_dense_layer(layer_idx)
            if layer.__class__.__name__ == "Conv2D":
                if self.logger:
                    self.logger.info(f"Quantizing (Conv2D) layer {layer_idx}...")
                self.quantize_conv2D_layer(layer_idx)
