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
from time import time

# The code in this script is generously provided by Caleb Robinson from his github repo which
# recreates the results of pretrained models on ImageNet. You can find a blogpost and a link
# to that repo on Caleb's website.
# https://calebrob.com/ml/imagenet/ilsvrc2012/2018/10/22/imagenet-benchmarking.html.
#
# This script shards and resizes images to 224x224. Importantly, it does not preprocess the images
# according to pretrained model specifications. This is because the pretrained models all
# preprocess in different manners, so that is left to the script which evaluates the networks on
# the test data.

def humansize(nbytes):
    '''From https://stackoverflow.com/questions/14996453/python-libraries-to-calculate-human-readable-filesize-from-bytes'''
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def load_images(image_paths, returned_shard, n_shards=5):
    """ loads images into memory. It only load and returns images of the 'returned_shard'.
        image_paths: a list of paths to images
        n_shards: number of shards to loaded images be divided.
        returned_shard: the part of images to be returned. 0 <= returned_shard < n_shards
    """
    assert 0 <= returned_shard < n_shards, "The argument returned_shard must be between 0 and n_shards"
    
    shard_size = len(image_paths) // n_shards
    sharded_image_paths = image_paths[returned_shard*shard_size:(returned_shard+1)*shard_size] if returned_shard < n_shards - 1 \
                     else image_paths[returned_shard*shard_size:]
    
    images_list = np.zeros((len(sharded_image_paths), 224, 224, 3), dtype=np.uint8)
    
    for i, image_path in enumerate(sharded_image_paths):
        
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
        

        images_list[i, ...] = image[..., ::-1]
        
    return images_list

def shard_images(n_shards=500):

    for i in range(n_shards):
        
        images = load_images(image_paths, returned_shard=i, n_shards=n_shards)
        
        if i == 0:
            print("Total memory allocated for loading images:", humansize(images.nbytes))
        
        np.save(str(path_imagenet_val_dataset / "x_val_{}.npy".format(i+1)), images)
        
        if (i + 1) * 100 / n_shards % 5 == 0:
            print("{:.0f}% Completed.".format((i+1)/n_shards*100))
        
        images = None

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
        

        return image[..., ::-1]

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

    np.save(str(path_imagenet_val_dataset/"y_val.npy"), y_val)
    return y_val

def _clean_up_train_test_split():
    # Migrate nested training image directory structure back to unstructured validation directory.
    if path_train_images.exists():
        print("Existing training image population exists. Cleaning up...", end="")
        nested_image_paths = glob(str(path_train_images/"*/*.npy"))
        for image_path in nested_image_paths:
            os.remove(image_path)
        # Get rid of the now empty training directory.
        for nested_dir in os.listdir(path_train_images):
            os.rmdir(f"{path_train_images}/{nested_dir}")
        os.rmdir(path_train_images)
        print("done.")

    # Migrate nested test image directory structure back to unstructured validation directory.
    if path_test_images.exists():
        print("Existing test image population exists. Cleaning up...", end="")
        nested_image_paths = glob(str(path_test_images/"*/*.npy"))
        for image_path in nested_image_paths:
            os.remove(image_path)
        # Get rid of the now empty test directory
        for nested_dir in os.listdir(path_test_images):
            os.rmdir(f"{path_test_images}/{nested_dir}")
        os.rmdir(path_test_images)
        print("done.")

def _add_image_to_dir(image_path, label, parent_dir):

    # Grab the filename of this image without the path or the file extension.
    image_filename =  image_path.split('/')[-1].split('.')[-2]
    # Initialize what the nested subdirectory's name is.
    image_directory = str(parent_dir) + f"/{label}"
    # Determine if you need to create a nested subdirectory for this new image.
    if str(label) not in os.listdir(parent_dir):
        os.mkdir(image_directory)

    # Preprocess image, then save preprocessed image into labelled subdirectory.
    image = _preprocess_image(image_path)
    new_imagepath_name = str(image_directory) + "/preprocessed_" + str(image_filename)
    np.save(new_imagepath_name, image)
    # shutil.copy(image_path, image_directory)

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

    path_imagenet_val_dataset = Path("../data/") # path/to/data/
    path_train_images = Path("../train_data/")
    path_test_images = Path("../test_data/")
    dir_images = Path("../data/val") # path/to/images/directory
    path_labels = Path("../data/ILSVRC2012_validation_ground_truth.txt")
    path_synset_words = Path("../data/synset_words.txt")
    path_meta = Path("../data/meta.mat")

    # If an existing split is present, then delete it.
    _clean_up_train_test_split()

    image_paths = sorted(glob(str(dir_images/"*")))
    n_images = len(image_paths)

    # Grab the command line input for number of images to use for quantization training.
    quantization_training_size = int(sys.argv[1])
    test_size = n_images - quantization_training_size
    if test_size < 0:
        raise ValueError(f"{quantization_training_size} images were requested to be used for learning quantization, but there are only {n_images} images in total.")

    meta = scipy.io.loadmat(str(path_meta))
    original_idx_to_synset = {}
    synset_to_name = {}


    # Shard the images for quicker I/O.
    # shard_images()

    # Use the devkit to generate labels for validation data
    labels = generate_labels()

    try:
        tic = time()
        print(f"Preprocessing and splitting training and test populations...")
        train_test_split(labels)
        print(f"done preprocessing and splitting. {time() - tic:.2f} seconds.")
    except Exception as exc:
        print(f"An error occured in splitting. Cleaning up partial split.")
        _clean_up_train_test_split()
        print("done with clean up.")
        raise(exc)



