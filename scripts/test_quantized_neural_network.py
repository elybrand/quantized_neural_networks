import numpy as np
import pytest
from collections import namedtuple
from quantized_network import QuantizedNeuralNetwork
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Sequential

RandomWalk = namedtuple('RandomWalk', ['w', 'X'])

@pytest.fixture
def parallel_random_walk():
	w = np.array([1,1,1])
	X = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    return RandomWalk(w=w, X=X)


@pytest.fixture
def adversarial_ortho_walk():
	w = np.array([0.5, 0.5, 0.5, 0.5])
	X = np.array([[1, 0, 0, 1/np.sqrt(3)], [0, 1, 0, 1/np.sqrt(3)], [0, 0, 1, -1/np.sqrt(3)]])
    return RandomWalk(w=w, X=X)

def test_select_next_directions(adversarial_ortho_walk):

	w, X = adversarial_ortho_walk.w, adversarial_ortho_walk.X
	# model = Sequential()
	# model.add(Dense(1, activation=None, use_bias=False, input_dim=(3,)))
	# model.layers[0].set_weights([w])
	qnet = QuantizedNeuralnetwork(
		network=None,
	    batch_size=None,
	    get_data=None,
    )
    u = X[:,0]
	qnet.select_next_directions(w, u, X[:,1:], X[:,1:])

