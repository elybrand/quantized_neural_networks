import numpy as np
import pandas as pd
import logging
from itertools import product
from collections import namedtuple
from tensorflow import argsort, cast, transpose, argmax, int32, float32
from tensorflow.math import reduce_any, reduce_mean
from tensorflow.random import set_seed
from tensorflow.keras.models import load_model, clone_model, save_model
from tensorflow.keras.applications import ResNet50, MobileNet
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.utils import to_categorical
from quantized_network import QuantizedCNN, ImageNetSequence
from itertools import chain
from time import time
from sys import stdout, argv
from os import mkdir, remove
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

# Specify the paths that contain the quantization training data and test data.
# Note that these directories are formed *after* running the preprocess_imagenet.py script.
dir_imagenet_val_dataset = Path("../data/")
dir_processed_images = Path("../data/preprocessed_val/")

# Grab the pretrained model name
pretrained_model = [MobileNet]
preprocess_func = [mobilenet_preprocess_input]
data_sets = ["ILSVRC2012"]
q_train_sizes = [25000]
bits = [np.log2(i) for i in  (16,)]
alphabet_scalars = [2, 3, 4, 5, 6]

np_seed = 0
tf_seed = 0

parameter_grid = product(
    pretrained_model,
    preprocess_func,
    data_sets,
    q_train_sizes,
    bits,
    alphabet_scalars,
)

ParamConfig = namedtuple(
    "ParamConfig",
    "pretrained_model, preprocess_func, data_set, q_train_size, bits, alphabet_scalar",
)
param_iterable = (ParamConfig(*config) for config in parameter_grid)

def _preprocess_and_reshape(image_path, preprocess_func):
    image = np.load(image_path)
    preprocessed_image = preprocess_func(image)
    reshaped_image = np.reshape(preprocessed_image, (1, *preprocessed_image.shape))
    return reshaped_image

def get_image_generator(image_paths, preprocess_func, epochs=1):
    image_iterator = (_preprocess_and_reshape(image_path, preprocess_func) for image_path in image_paths)
    for epoch in range(1, epochs):
        # Chain together another copy of base_iterator if epochs > 1.
        image_iterator = chain(image_iterator, (_preprocess_and_reshape(image_path, preprocess_func) for image_path in image_paths))

    return image_iterator

def top_k_accuracy(y_true, y_pred, k=1, tf_enabled=True):
    '''
    Calculates top_k accuracy of predictions. Expects both y_true and y_pred to be one-hot encoded.
    numpy implementation is from: https://github.com/chainer/chainer/issues/606
    '''

    if tf_enabled:
        argsorted_y = argsort(y_pred)[:,-k:]
        matches = cast(reduce_any(transpose(argsorted_y) == argmax(y_true, axis=1, output_type=int32), axis=0), float32)
        return  reduce_mean(matches).numpy()
    else:
        argsorted_y = np.argsort(y_pred)[:,-k:]
        return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()

def quantize_network(parameters: ParamConfig) -> pd.DataFrame:

    # Reset the seeds for splitting training and testing
    np.random.seed(np_seed)
    set_seed(tf_seed)

    # Load the image paths and the labels. Order of labels must match
    # alphanumeric sorting of the paths, so we sort the paths.
    image_paths = np.array(sorted(glob(str(dir_processed_images/"*.npy"))))
    num_images = len(image_paths)
    y = np.load(str(dir_imagenet_val_dataset/"y_val.npy"))

    train_idxs = np.random.choice(range(num_images), size=parameters.q_train_size, replace=False,)
    test_idxs = list(set(range(len(image_paths))).difference(set(train_idxs)))

    train_paths = image_paths[train_idxs]
    test_paths = image_paths[test_idxs]
    y_train = y[train_idxs]
    y_test = y[test_idxs]

    # Use one-hot encoding for labels. 
    num_classes = np.unique(y).shape[0]
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Load the model with imagenet weights and the top layer included.
    model = parameters.pretrained_model()
    model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    logger.info("Generating analog predicted labels...")
    tic = time()
    test_generator = ImageNetSequence(test_paths, y_test, batch_size=32, preprocess_func=parameters.preprocess_func)
    y_test_pred_analog = model.predict(test_generator, verbose=True)
    logger.info(f"done. {time()-tic:.2f} seconds.")
    top1_analog = top_k_accuracy(y_test, y_test_pred_analog, k=1, tf_enabled=True)
    top5_analog = top_k_accuracy(y_test, y_test_pred_analog, k=5, tf_enabled=True)

    # train_generator = get_image_generator(train_paths, parameters.preprocess_func, epochs=1)
    # y_train_pred_analog = model.predict(train_generator, verbose=True)
    # top1_analog = top_k_accuracy(y_train, y_train_pred_analog, k=1, tf_enabled=True)
    # top5_analog = top_k_accuracy(y_train, y_train_pred_analog, k=5, tf_enabled=True)

    logger.info(f"Analog network (top 1 accuracy, top 5 accuracy) = ({top1_analog:.2f}, {top5_analog:.2f})")


    # Find out how many layers you're going to quantize. This tells us how many
    # times we need to chain together the training image generator.
    layer_names = np.array([layer.__class__.__name__ for layer in model.layers])
    num_layers_to_quantize = sum((layer_names == 'Dense') + (layer_names == 'Conv2D'))
    quantization_train_generator = get_image_generator(train_paths, parameters.preprocess_func, epochs=num_layers_to_quantize)

    my_quant_net = QuantizedCNN(
        network=model,
        batch_size=parameters.q_train_size,
        get_data=quantization_train_generator,
        logger=logger,
        bits=parameters.bits,
        alphabet_scalar=parameters.alphabet_scalar,
    )
    try:
        tic = time()
        my_quant_net.quantize_network()
        quantization_time = time()-tic
    except:
        # Probably ran out of disk space. Clean up the patch tensors and write to logger.
        logger.info(f"An exception occurred. Cleaning up patch tensors...")
        patch_tensor_paths = sorted(glob("./*.h5"))
        for path in patch_tensor_paths:
            remove(path)

    my_quant_net.quantized_net.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Serialize the greedy network.
    model_timestamp = str(pd.Timestamp.now()).replace(" ", "_").replace(":","").replace(".","")
    model_name = f"quantized_{model.name}_scaler{parameters.alphabet_scalar}_{parameters.bits}bits_{model_timestamp}"
    model_name = model_name.replace(".","")
    save_model(my_quant_net.quantized_net, f"../quantized_models/{model_name}")

    test_generator = test_generator = ImageNetSequence(test_paths, y_test, batch_size=32, preprocess_func=parameters.preprocess_func)
    y_test_pred_gpfq = my_quant_net.quantized_net.predict(test_generator, verbose=True)
    top1_gpfq = top_k_accuracy(y_test, y_test_pred_gpfq, k=1, tf_enabled=True)
    top5_gpfq = top_k_accuracy(y_test, y_test_pred_gpfq, k=5, tf_enabled=True)

    # train_generator = get_image_generator(train_paths, parameters.preprocess_func, epochs=1)
    # y_train_pred_gpfq = my_quant_net.quantized_net.predict(train_generator, verbose=True)
    # top1_gpfq = top_k_accuracy(y_train, y_train_pred_gpfq, k=1, tf_enabled=True)
    # top5_gpfq = top_k_accuracy(y_train, y_train_pred_gpfq, k=5, tf_enabled=True)

    logger.info(f"GPFQ network (top 1 accuracy, top 5 accuracy) = ({top1_gpfq:.2f}, {top5_gpfq:.2f})")

    logger.info(f"Building MSQ network...")
    tic = time()
    # Construct MSQ Net.
    MSQ_model = clone_model(model)
    # Set all the weights to be equal at first. This matters for batch normalization layers.
    MSQ_model.set_weights(model.get_weights())
    for layer_idx, layer in enumerate(model.layers):
        if layer.__class__.__name__ in ("Dense", "Conv2D"):
            # Use the same radius as the alphabet in the corresponding layer of the GPFQ network.
            rad = max(
                my_quant_net.quantized_net.layers[layer_idx].get_weights()[0].flatten()
            )
            if MSQ_model.layers[layer_idx].use_bias:
                W, b = model.layers[layer_idx].get_weights()
            else:
                # There's no bias vector
                W = model.layers[layer_idx].get_weights()[0]
            Q = np.array([my_quant_net._bit_round(w, rad) for w in W.flatten()]).reshape(
                W.shape
            )
            if MSQ_model.layers[layer_idx].use_bias:
                MSQ_model.layers[layer_idx].set_weights([Q, b])
            else:
                MSQ_model.layers[layer_idx].set_weights([Q])
    logger.info(f"done. {time()-tic:.2f} seconds.")
    MSQ_model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    test_generator = test_generator = ImageNetSequence(test_paths, y_test, batch_size=32, preprocess_func=parameters.preprocess_func)
    y_test_pred_msq = MSQ_model.predict(test_generator, verbose=True)
    top1_msq = top_k_accuracy(y_test, y_test_pred_msq, k=1, tf_enabled=True)
    top5_msq = top_k_accuracy(y_test, y_test_pred_msq, k=5, tf_enabled=True)

    # train_generator = get_image_generator(train_paths, parameters.preprocess_func, epochs=1)
    # y_train_pred_msq = MSQ_model.predict(train_generator, verbose=True)
    # top1_msq = top_k_accuracy(y_train, y_train_pred_msq, k=1, tf_enabled=True)
    # top5_msq = top_k_accuracy(y_train, y_train_pred_msq, k=5, tf_enabled=True)

    logger.info(f"MSQ network (top 1 accuracy, top 5 accuracy) = ({top1_msq:.2f}, {top5_msq:.2f})")

    trial_metrics = pd.DataFrame(
        {
            "data_set": parameters.data_set,
            "serialized_model": model.name,
            "q_train_size": parameters.q_train_size,
            "bits": parameters.bits,
            "alphabet_scalar": parameters.alphabet_scalar,
            "analog_test_top1_acc": top1_analog,
            "analog_test_top5_acc": top5_analog,
            "gpfq_test_top1_acc": top1_gpfq,
            "gpfq_test_top5_acc": top5_gpfq,
            "msq_test_top1_acc": top1_msq,
            "msq_test_top5_acc": top5_msq,
            "quantization_time": quantization_time,
            "np_seed": np_seed,
            "tf_seed": tf_seed,
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