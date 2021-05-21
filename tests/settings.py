from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.initializers import Constant as constant_kernel
from tensorflow.keras.layers import Dense
from tensorflow import constant

class TestSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):

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

INPUT_SHAPE = (2,)
LAYER_WIDTHS = (3, 2)
ACTIVATION = "linear"
USE_BIAS = False
INPUT_LAYER = Dense(
            LAYER_WIDTHS[0],
            input_shape=INPUT_SHAPE,
            activation=ACTIVATION,
            kernel_initializer=constant_kernel(value=1),
            use_bias=USE_BIAS,
        	)
HIDDEN_LAYER = Dense(
            LAYER_WIDTHS[1],
            activation=ACTIVATION,
            kernel_initializer=constant_kernel(value=1),
            use_bias=USE_BIAS,
        	)
FEEDFORWARD_NET = Sequential()
FEEDFORWARD_NET.add(INPUT_LAYER)
FEEDFORWARD_NET.add(HIDDEN_LAYER)

DATA = constant([ [1, 0], [0, 2] ])
LABELS = constant([ [3], [6] ])
DATA_GENERATOR = TestSequence(DATA, LABELS, batch_size=1)