import numpy as np
import pandas as pd
import logging
from itertools import product
from collections import namedtuple
from tensorflow.random import set_seed
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import load_model, clone_model, save_model
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.preprocessing.image import load_img # Very useful post for pretrained model on imagenet https://learnopencv.com/keras-tutorial-using-pre-trained-imagenet-models/
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from quantized_network import QuantizedCNN
from PIL import Image
from itertools import chain
from sys import stdout, argv
from os import mkdir

# Write logs to file and to stdout. Overwrite previous log file.
fh = logging.FileHandler("../train_logs/model_quantizing.log", mode="w+")
fh.setLevel(logging.INFO)
sh = logging.StreamHandler(stream=stdout)
sh.setLevel(logging.INFO)

# Only use the logger in this module.
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

# Load pretrained model from tensorflow
pretrained_model = MobileNetV2(weights="imagenet")

q_train_sizes = [100]
ignore_layers = [[]]
bits = [4]
alphabet_scalars = [2]

parameter_grid = product(
    pretrained_model,
    q_train_sizes,
    ignore_layers,
    bits,
    alphabet_scalars,
)

ParamConfig = namedtuple(
    "ParamConfig",
    "pretrained_model, q_train_size, ignore_layers, bits, alphabet_scalar",
)
param_iterable = (ParamConfig(*config) for config in parameter_grid)

def quantize_network(parameters: ParamConfig) -> pd.DataFrame:

    # Consider using ImageDataGenerator. See https://towardsdatascience.com/transfer-learning-in-action-from-imagenet-to-tiny-imagenet-b96fe3aa5973

    # assign the image path for the classification experiments
    filename = ...
    # load an image in PIL format
    original = load_img(filename, target_size=(224, 224))

    # convert the PIL image to a numpy array
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)

    processed_image = pretrained_model.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = pretrained_model.predict(processed_image)
    # convert the probabilities to class labels
    # we will get top 5 predictions which is the default
    label_output = decode_predictions(predictions)

    # Calculate pretrained model accuracy on test data
    analog_loss, analog_accuracy = pretrained_model.evaluate(X_test, y_test, verbose=True)
    logger.info(f"Analog network test accuracy = {analog_accuracy:.2f}")

    # Find out how many layers you're going to quantize.
    layer_names = np.array([layer.__class__.__name__ for layer in pretrained_model.layers])
    num_layers_to_quantize = sum((layer_names == 'Dense') + (layer_names == 'Conv2D'))

    get_data = (sample for sample in X_train[0:quant_train_size])
    for i in range(num_layers_to_quantize):
        # Chain together iterators over the entire training set. This is so each layer uses
        # the entire training data.
        get_data = chain(get_data, (sample for sample in X_train[0:quant_train_size]))
    batch_size = quant_train_size

    my_quant_net = QuantizedCNN(
        network=pretrained_model,
        batch_size=batch_size,
        get_data=get_data,
        logger=logger,
        bits=parameters.bits,
        alphabet_scalar=parameters.alphabet_scalar,
    )
    my_quant_net.quantize_network()
    my_quant_net.quantized_net.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )


    # Serialize the greedy network. Get rid of problematic characters for the filename.
    model_timestamp = str(pd.Timestamp.now()).replace(" ", "_").replace(":","").replace(".","").replace("/","")
    model_name = f"quantized_imagenet_scaler{parameters.alphabet_scalar}_{model_timestamp}"
    save_model(my_quant_net.quantized_net, f"../quantized_models/{model_name}")

    q_loss, q_accuracy = my_quant_net.quantized_net.evaluate(X_test, y_test, verbose=True)

    # Construct MSQ Net.
    MSQ_model = clone_model(model)
    # Set all the weights to be equal at first. This matters for batch normalization layers.
    MSQ_model.set_weights(model.get_weights())
    for layer_idx, layer in enumerate(model.layers):
        if layer.__class__.__name__ in ("Dense", "Conv2D"):
            # Use the same radius as the alphabet in the corresponding layer of the Sigma Delta network.
            rad = max(
                my_quant_net.quantized_net.layers[layer_idx].get_weights()[0].flatten()
            )
            W, b = pretrained_model.layers[layer_idx].get_weights()
            Q = np.array([my_quant_net._bit_round(w, rad) for w in W.flatten()]).reshape(
                W.shape
            )
            MSQ_model.layers[layer_idx].set_weights([Q, b])

    MSQ_model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    MSQ_loss, MSQ_accuracy = MSQ_model.evaluate(X_test, y_test, verbose=True)

    trial_metrics = pd.DataFrame(
        {
            "data_set": 'imagenet',
            "pretrained_model": pretrained_model.name,
            "serialized_model": parameters.pretrained_model,
            "q_train_size": parameters.q_train_size,
            "ignore_layers": [parameters.ignore_layers],
            "bits": parameters.bits,
            "alphabet_scalar": parameters.alphabet_scalar,
            "analog_test_acc": analog_accuracy,
            "sd_test_acc": q_accuracy,
            "msq_test_acc": MSQ_accuracy,
        },
        index=[model_timestamp],
    )

    return trial_metrics


if __name__ == "__main__":

    # Store results in csv file.
    file_name = data_sets[0] + "_model_metrics_" + timestamp
    # Timestamp adds a space. Replace it with _
    file_name = file_name.replace(" ", "_")
    for idx, params in enumerate(param_iterable):
        trial_metrics = quantize_network(params)

        if idx == 0:
            # add the header
            trial_metrics.to_csv(f"../model_metrics/{file_name}.csv", mode="a")
        else:
            trial_metrics.to_csv(
                f"../model_metrics/{file_name}.csv", mode="a", header=False
            )