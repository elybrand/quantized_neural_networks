import numpy as np
from collections import namedtuple
from quantized_network import QuantizedNeuralNetwork
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Sequential

RandomWalk = namedtuple("RandomWalk", ["w", "X"])


@pytest.fixture
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


def test_select_next_directions(adversarial_ortho_walk):

    # w, X = adversarial_ortho_walk.w, adversarial_ortho_walk.X
    w = np.array([[0.5], [0.5], [0.5]])
    X = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]).T

    # in R2, try using unit vectors along the unit circle. Fix a permutation to shuffle them.
    # Sort, and make sure you get the permutation back.

    model = Sequential()
    model.add(Dense(1, activation=None, use_bias=False, input_dim=3))
    model.layers[0].set_weights([w])
    qnet = QuantizedNeuralNetwork(network=model, batch_size=None, get_data=None,)
    sdirs = qnet.sort_directions(wX=X, qX=X)
    u = X[:, 0]
    qnet.select_next_directions(w, u, X[:, 1:], X[:, 1:])


def test_sort_directions():

    w = np.array([[0.1], [0.2], [0.3], [0.4]])
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]]).T

    perm = (2, 0, 3, 1)
    w_perm = np.array([w[idx] for idx in perm])
    X_perm = np.array([X.T[idx, :] for idx in perm]).T

    batch_size = 2
    get_data = (sample for sample in X)

    model = Sequential()
    model.add(Dense(1, activation=None, use_bias=False, input_dim=4))
    model.layers[0].set_weights([w_perm])
    qnet = QuantizedNeuralNetwork(
        network=model, batch_size=batch_size, get_data=get_data,
    )
    qnet.quantize_network()
