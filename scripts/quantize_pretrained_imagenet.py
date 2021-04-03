import numpy as np
import pandas as pd
import logging
from itertools import product
from collections import namedtuple
from tensorflow.random import set_seed
from tensorflow.keras.models import load_model, clone_model, save_model
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedCNN
from itertools import chain
from time import time
from sys import stdout, argv
from os import mkdir
from pathlib import Path
from glob import glob
import scipy.io
import cv2

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

# Grab the pretrained model name
pretrained_model = [ResNet50]
data_sets = ["ILSVRC2012"]
q_train_sizes = [1000]
bits = [np.log2(i) for i in  (3, 4, 8, 16)]
alphabet_scalars = [2, 3, 4, 5, 6]

parameter_grid = product(
    pretrained_model,
    data_sets,
    q_train_sizes,
    bits,
    alphabet_scalars,
)

ParamConfig = namedtuple(
    "ParamConfig",
    "pretrained_model, data_set, q_train_size, bits, alphabet_scalar",
)
param_iterable = (ParamConfig(*config) for config in parameter_grid)

def quantize_network(parameters: ParamConfig) -> pd.DataFrame:

    # Consider using ImageDataGenerator. See https://towardsdatascience.com/transfer-learning-in-action-from-imagenet-to-tiny-imagenet-b96fe3aa5973
    # You'll want to use .flow_from_directory(). Use class_mode=None, and you'll still need the data
    # to be in a subdirectory.

    # TODO: the ImageDataGenerator class is nice, but it does assume a special directory
    # structure. You need to somehow pair images with labels in a way that makes it
    # easier to divide into train and test. Caleb's script doesn't account for this split.

    # Initialize generator objects with the correct preprocessing function
    # for yielding the training and test data.
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=parameters.pretrained_model.preprocess_input,
    )
    test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=parameters.pretrained_model.preprocess_input,
    )

    # Tell the generators from which directories to pull from. 
    train_get_data = image_generator.flow_from_directory(
                    directory, # TODO
                    target_size=(224, 224), # This matches the dimensions that Caleb's script resizes to.
                    class_mode=None, # We do not need labels for learning the quantization.
                    batch_size=1,    # The quantized network classes request one at a time.
                    shuffle=True,    # TODO: This shuffle is problematic because Caleb's script
                                     # saves labels according to how the data are ordered.
                    seed=0,          # Set a random seed for shuffling.
                    interpolation="nearest", # Define the upsampling method in case dimensions don't match.
                                             # Caleb's script already resizes the images to 224x224, so we
                                             # don't need to worry about this kwarg.
                )

    # TODO: I don't have the directory structure set up to infer class labels here.
    # Also, you can do this once and for all outside of the quantize_network() routine
    # in case that saves you time.
    test_get_data = image_generator.flow_from_directory(
                directory, # TODO
                target_size=(224, 224), # This matches the dimensions that Caleb's script resizes to.
                class_mode=None,    # We do not need labels for learning the quantization.
                batch_size=32, 
                shuffle=True,       # TODO: This shuffle is problematic because Caleb's script
                                    # saves labels according to how the data are ordered.
                seed=0,             # Set a random seed for shuffling.
                interpolation="nearest", # Define the upsampling method in case dimensions don't match.
                                         # Caleb's script already resizes the images to 224x224, so we
                                         # don't need to worry about this kwarg.
            )

    num_classes = np.unique(y_train).shape[0]
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    input_shape = X_train[0].shape

    # Load the model.
    model = parameters.pretrained_model
    # TODO: feed in image_generators.
    analog_loss, analog_accuracy = model.evaluate(X_test, y_test, verbose=True)
    logger.info(f"Analog network test accuracy = {analog_accuracy:.2f}")

    # Find out how many layers you're going to quantize.
    layer_names = np.array([layer.__class__.__name__ for layer in model.layers])
    num_layers_to_quantize = sum((layer_names == 'Dense') + (layer_names == 'Conv2D'))

    get_data = (sample for sample in X_train[0:quant_train_size])
    for i in range(num_layers_to_quantize):
        # Chain together iterators over the entire training set. This is so each layer uses
        # the entire training data.
        get_data = chain(get_data, (sample for sample in X_train[0:quant_train_size]))
    batch_size = quant_train_size

    my_quant_net = QuantizedCNN(
        network=model,
        batch_size=batch_size,
        get_data=get_data,
        logger=logger,
        bits=parameters.bits,
        alphabet_scalar=parameters.alphabet_scalar,
    )
    tic = time()
    my_quant_net.quantize_network()
    quantization_time = time()-tic

    # TODO: look at both top-1 and top-5.
    my_quant_net.quantized_net.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Serialize the greedy network.
    model_timestamp = str(pd.Timestamp.now()).replace(" ", "_").replace(":","").replace(".","")
    model_name = f"quantized_{model.name}_scaler{parameters.alphabet_scalar}_{parameters.bits}bits_{model_timestamp}"
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
            W, b = model.layers[layer_idx].get_weights()
            Q = np.array([my_quant_net._bit_round(w, rad) for w in W.flatten()]).reshape(
                W.shape
            )
            MSQ_model.layers[layer_idx].set_weights([Q, b])

    # TODO: look at both top-1 and top-5.
    MSQ_model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    MSQ_loss, MSQ_accuracy = MSQ_model.evaluate(X_test, y_test, verbose=True)

    trial_metrics = pd.DataFrame(
        {
            "data_set": parameters.data_set,
            "serialized_model": parameters.pretrained_model.name,
            "q_train_size": parameters.q_train_size,
            "bits": parameters.bits,
            "alphabet_scalar": parameters.alphabet_scalar,
            "analog_test_top1_acc": ,# TODO
            "analog_test_top5_acc": ,# TODO
            "sd_test_top1_acc": , #TODO
            "sd_test_top5_acc": , #TODO
            "msq_test_top1_acc": , #TODO
            "msq_test_top5_acc": , #TODO
            "quantization_time": quantization_time,
        },
        index=[model_timestamp],
    )

    return trial_metrics


if __name__ == "__main__":

    # Store results in csv file.
    timestamp = str(pd.Timestamp.now()).replace(" ", "_").replace(":","").replace(".","")
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