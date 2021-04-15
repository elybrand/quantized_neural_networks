import sys, os, time
from pathlib import Path
from glob import glob

# Select GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import scipy.io
import cv2
import shutil
import concurrent.futures
from time import time

# The code in this script is generously provided by Caleb Robinson from his github repo which
# recreates the results of pretrained models on ImageNet. You can find a blogpost and a link
# to that repo on Caleb's website.
# https://calebrob.com/ml/imagenet/ilsvrc2012/2018/10/22/imagenet-benchmarking.html.
#
# This script applies baseline preprocessing for all pretrained models, and resizes images to 224x224. 
# Importantly, however, it does not further preprocess the images according to pretrained model specifications. 
# This is because the pretrained models all preprocess in different manners, so that is left to the script 
# which evaluates the networks on the test data.

dir_imagenet_val_dataset = Path("../data/") # path/to/data/
dir_images = Path("../data/val") # path/to/images/directory
dir_processed_images = Path("../data/preprocessed_val/")
path_labels = Path("../data/ILSVRC2012_validation_ground_truth.txt")
path_synset_words = Path("../data/synset_words.txt")
path_meta = Path("../data/meta.mat")

def _clean_up_processed_images():
    # Migrate nested training image directory structure back to unstructured validation directory.
    if dir_processed_images.exists():
        shutil.rmtree(dir_processed_images)

def _preprocess_image(image_path):

        # Load (in BGR channel order)
        image = cv2.imread(image_path)
        
        # Resize
        height, width, _ = image.shape
        new_height = height * 256 // min(image.shape[:2])
        new_width = width * 256 // min(image.shape[:2])
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Crop
        height, width, _ = image.shape
        startx = width//2 - (224//2)
        starty = height//2 - (224//2)
        image = image[starty:starty+224,startx:startx+224]
        assert image.shape[0] == 224 and image.shape[1] == 224, (image.shape, height, width)
        

        # Grab the name of the image file without the file extension.
        filename = image_path.split('/')[-1].split('.')[0]
        new_filename = "preprocessed_" + filename + ".npy"
        new_image_path = str(dir_processed_images) + "/" + new_filename

        # Save the image to disk, making sure to reorder channels into RGB order.
        np.save(new_image_path, image[..., ::-1])

def generate_labels():
    for i in range(1000):
        ilsvrc2012_id = int(meta["synsets"][i,0][0][0][0])
        synset = meta["synsets"][i,0][1][0]
        name = meta["synsets"][i,0][2][0]
        original_idx_to_synset[ilsvrc2012_id] = synset
        synset_to_name[synset] = name

    synset_to_keras_idx = {}
    keras_idx_to_name = {}
    with open(str(path_synset_words), "r") as f:
        for idx, line in enumerate(f):
            parts = line.split(" ")
            synset_to_keras_idx[parts[0]] = idx
            keras_idx_to_name[idx] = " ".join(parts[1:])

    convert_original_idx_to_keras_idx = lambda idx: synset_to_keras_idx[original_idx_to_synset[idx]]

    with open(str(path_labels),"r") as f:
        y_val = f.read().strip().split("\n")
        y_val = np.array([convert_original_idx_to_keras_idx(int(idx)) for idx in y_val])

    np.save(str(dir_imagenet_val_dataset/"y_val.npy"), y_val)

def train_test_split(labels):

    n_images = labels.size

    # Make subdirectories which separate the quantization training data and the test data.
    os.mkdir(path_train_images)
    os.mkdir(path_test_images)

    # Randomly select images to be used for training.
    training_image_paths = set(np.random.choice(image_paths, size=quantization_training_size, replace=False))
    test_image_paths = set(image_paths)-training_image_paths

    # Go through the validation set and put them in nested subdirectories of either the training
    # or test directory. The nested directory should be the *Keras* label of the image.

    # TODO (low priority): This could be multiprocessed.
    for idx, image_path in enumerate(image_paths):
        # Determine if this image is in test or training image.
        is_test_image = image_path in test_image_paths
        if is_test_image:
            _add_image_to_dir(image_path, labels[idx], path_test_images)
        else:
            _add_image_to_dir(image_path, labels[idx], path_train_images)

        if (idx + 1) * 100 / n_images % 5 == 0:
            print("\t{:.0f}% Completed.".format((idx+1)/n_images*100))

if __name__ == "__main__":

    image_paths = sorted(glob(str(dir_images/"*")))
    n_images = len(image_paths)
    meta = scipy.io.loadmat(str(path_meta))
    original_idx_to_synset = {}
    synset_to_name = {}

    _clean_up_processed_images()

    os.mkdir(dir_processed_images)

    # Use the devkit to generate labels for validation data
    generate_labels()
    print("Preprocessing data...")
    tic = time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_path = {executor.submit(_preprocess_image, image_path): image_path for image_path in image_paths}
        for future in concurrent.futures.as_completed(future_to_path):
                future.result()
    print(f"done. {time() - tic:.2f} seconds.")



