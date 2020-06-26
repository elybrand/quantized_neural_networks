import numpy as np
from collections import namedtuple
from quantized_network import QuantizedNeuralNetwork
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Sequential

RandomWalk = namedtuple("RandomWalk", ["w", "X"])


def parallel_random_walk():
    w = np.array([1, 1, 1])
    X = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    return RandomWalk(w=w, X=X)


@pytest.fixture
def adversarial_ortho_walk():
    w = np.array([0.5, 0.5, 0.5, 0.5])
    X = np.array(
        [[1, 0, 0, 1 / np.sqrt(3)], [0, 1, 0, 1 / np.sqrt(3)], [0, 0, 1, -1 / np.sqrt(3)]]
    )
    return RandomWalk(w=w, X=X)


def test_sort_directions():

    w = np.array([[0.1], [0.2], [0.3], [0.4]])
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]]).T

    perm = (0, 2, 1, 3)
    w_perm = np.array([w[idx] for idx in perm])
    X_perm = np.array([X.T[idx, :] for idx in perm]).T

    batch_size = 2
    get_data = (sample for sample in X_perm)

    model = Sequential()
    model.add(Dense(1, activation=None, use_bias=False, input_dim=4))
    model.layers[0].set_weights([w_perm])
    qnet = QuantizedNeuralNetwork(
        network=model, batch_size=batch_size, get_data=get_data,
    )
    sdirs = qnet.sort_directions(wX=X_perm, qX=X_perm)
    qnet.quantize_network(use_greedy=True)
    qnet.quantized_net.layers[0].get_weights()  # @pytest.fixture


def test_mp_quantization():

    w = np.array([[1], [3], [7]])
    X = np.array([[2, 0], [1, 0], [0, 4]]).T

    batch_size = 2
    get_data = (sample for sample in X)

    model = Sequential()
    model.add(Dense(1, activation=None, use_bias=False, input_dim=3))
    model.layers[0].set_weights([w])
    qnet = QuantizedNeuralNetwork(
        network=model, batch_size=batch_size, get_data=get_data,
    )
    qnet.quantize_network()
    # Bit string should be 3, 0, 3
    qnet.quantized_net.layers[0].get_weights()


def test():

    w = np.array([[0.886], [-0.8605], [-0.2612], [0.6923]])
    X = np.array([[1, 0], [0, 1], [1, 0], [0, 1]]).T

    batch_size = 2
    get_data = (sample for sample in X)

    model = Sequential()
    model.add(Dense(1, activation=None, use_bias=False, input_dim=4))
    model.layers[0].set_weights([w])
    qnet = QuantizedNeuralNetwork(
        network=model, batch_size=batch_size, get_data=get_data,
    )
    qnet.quantize_network()
    # Bit string should be 3, 0, 3
    qnet.quantized_net.layers[0].get_weights()
