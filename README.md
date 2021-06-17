# Quantized Neural Networks

This repository contains code for the experiments in the manuscript "[A Greedy Algorithm for Quantizing Neural Networks](https://arxiv.org/abs/2010.15979)" by Eric Lybrand and Rayan Saab (2020). These experiments include training and quantizing two networks: a multilayer perceptron to classify MNIST digits, and a convolutional neural network to classify CIFAR10 images. As of June 16, 2021 the repo also includes a script for quantizing the layers of pre-trained ImageNet networks using ImageNet data. I've included instructions for how to run these experiments in a container. If you have a virtual environment you'd like to use, or if you're willing to sort out the requirements listed in `requirements.txt` in your usual environment then you can feel free to skip to the relevant **NOTE** in each section with more concise instructions.

If you're only interested in instructions for recreating the ImageNet experiment, scroll towards the bottom of the README.

**NOTE**: When I began working on the ImageNet experiments, I migrated to a 2020 MacBook Pro with a M1 chip out of necessity. My old 2013 MacBook Pro was well beyond its limits with the CIFAR10 experiments. If your machine doesn't have GPU support for Tensorflow, then I hate to tell you but you're going to be waiting an eternity to recreate the ImageNet experiments. If you are using a machine with a M1 chip, you'll need to install [a special fork of Tensorflow](https://developer.apple.com/metal/tensorflow-plugin/). If you (like past me) are also having trouble finding compatible versions of pandas, scipy, and other libraries that use BLAS then let me save you the trouble and tell you that [you need to use the miniforge installer](https://towardsdatascience.com/new-m1-who-dis-677e085baffd).

## Setting Up Docker (Optional)
To run the code in a docker container, you'll first need to download [Docker](https://www.docker.com/get-started). Once you've got Docker installed, navigate to the github repo on your machine and run the following CLI command: `docker build --tag quant_nnets .` (don't forget the period). This will build a docker image with the python 3.7.10 base image and download other required python packages from `requirements.txt`. This is a somewhat large docker image since we have to download Tensorflow, but it's significantly smaller than the bloated AWS DLAMI I tried messing with. It may take a minute or two to build. To see that the docker image has built successefully, run the command `docker images` and look for `quant_nnets` under the repository column.

Another thing you'll have to do is to modify how much memory Docker Desktop is allotted. In Docker Destop, navigate to "Resources", then to "Advanced". I was able to run the scripts with 16GB memory allottment. This may be overkill, but I didn't bother to fine tune.

## Running Experiments
Once the image is built, you can start running the experiments. These experiments are set up so that model training and model compression occur in two separate scripts. Once we have a trained network, that network is saved in the directory `serialized_models`. To persist that trained model on your local machine, we'll use Docker volumes.

### Network Training (MNIST & CIFAR10)

To train a network on a docker container, run the following command
```
docker run -dit --name train_container \
                -v [absolute/path/to/repo]/serialized_models:/serialized_models \
                -v [absolute/path/to/repo]/train_logs:/train_logs \
           quant_nnets python [train_mnist_mlp.py, train_cifar10_cnn.py]
```

Importantly, to use Docker volumes you need to specify the absolute path to the github repo and place that where you see `[absolute/path/to/repo]` in the above command. You also should choose either `train_mnist_mlp.py` or `train_cifar10_cnn.py`  to run. Training the MNIST network should take only a few minutes, whereas the CIFAR10 network will take significantly longer. On my machine which didn't use any GPUS for training it took roughly 16 hours. 

**NOTE**: To run the scripts without instantiating a Docker contianer, just run the command `python [train_mnist_mlp.py, train_cifar10_cnn.py]` in the `scripts` subdirectory.

**NOTE**: If you're running the scripts in a container, the `-d` flag will run the container in the background, so you won't see any output onto the command line during model training. If you want to track the model training progress in real-time, you can replace the `-dit` flag with `-it`. This runs the docker container in attached mode. You will see the verbose output from the Keras training in your terminal. 

**NOTE**: The analog CIFAR10 network we used in our experiments is included in `serialized_models` if you don't want to train your own network. 

### Network Compression (MNIST & CIFAR10)
Once you have a pretrained network saved in `serialized_models` you're ready to run the network quantization script. Both model quantization scripts `quantize_pretrained_mlp.py` and `quantize_pretrained_cnn.py` define parameter grids at the top of their respective python files which they cross-validate over. Those parameter grids include:
1. choices for the number of training data to be used to learn the quantization,
2. the number of bits to use in the quantization alphabet, defined as `bits`,
3. the radius of the quantization alphabet, defined as `alphabet_scalars`.

In the paper, we only validate over the last two parameters. You're welcome to play around with the number of samples used to train the quantization. Just note that if you do make any modifications to the scripts, you'll need to rebuild the Docker container by again calling `docker build --tag quant_nnets .` Fortunately, Docker builds containers incrementally so you won't have to wait as long as you did the first time since all of the requirements will have already been downloaded.

**BEWARE** that training the quantization cannot be done with mini-batches of data! You will find that using the entire CIFAR10 dataset to train the quantization requires an enormous amount of disk space (think: for a convolutional layer with 256 filters, every image gets transformed from having 3 channels to 256 channels; further, instead of acting on the entire image the network acts on patches of data. That means the number of training images is not equal to the number of "samples" to learn the quantization; the number of samples is actually magnified by the number of patches and the number of channels). You'll also happen to find, like I did when I tried using the entire training data set, that you don't get that much better performance than using a much smaller subset. (As a matter of fact, you can actually get competitive performance only using ~1 training image to learn a 4 bit quantization for CIFAR10. Don't ask me to explain why. I wanted to throw this in the paper but it seems like a quirk and not a trend. Generally speaking, I found that the alphabet scalar and the bit budget were more influential on quantization test accuracy than the number of training samples.) I've set it up so that the training data for the quantization at each layer is saved to disk and is read one at a time to prevent running out of RAM if your local machine is constrained with RAM or if you really want to go overboard with the number of images.

To run the network quantization scripts on a particular serialzed model, run the following command
```
docker run -dit --name quant_container \
                -v [absolute/path/to/repo]/serialized_models:/serialized_models \
                -v [absolute/path/to/repo]/quantized_models:/quantized_models \
                -v [absolute/path/to/repo]/train_logs:/train_logs \
                -v [absolute/path/to/repo]/model_metrics:/model_metrics \
           quant_nnets python quantize_pretrained_[mlp, cnn].py [name of serialized model in serialized_models/]
```

While the quantization script is running, it will log the progress of the quantization to the file `train_logs/model_quantizing.log`. Once the network is fully quantized with a particular parameter configuration the script will do two things:
1. the parameters for that quantization and the test accuracies of the analog and quantized networks will be written to a .csv file in `model_metrics/`. This .csv file will store the results of all of the quantizations from one call to the script.
2. the quantized model will be serialized into the directory `quantized_models/`.

**NOTE**: To run the scripts without instantiating a Docker contianer, just run the command `python quantize_pretrained_[mlp, cnn].py [name of serialized model in serialized_models/]` in the `scripts` subdirectory.

**NOTE**: Running the quantization script for both MNIST and CIFAR10 networks takes a while because of the number of parameters to cross-validate over. For the MNIST data set, we also use half the training set (25000 images) which is far more than is needed to get a competitive quantization, and the layers are fairly wide. The median time it took to quantize a network on my machine for the MNIST network was around 5 minutes. The CIFAR10 network takes some time because the network is deep-ish, the code requires reformatting the data to "vectorize" the image patches, and we also need some OS calls to save the output of previous layers to disk so we don't run out of RAM. The median time it took to quantize a CIFAR10 network was around 30 minutes.

**NOTE**: As before, if you're running the scripts in a container then remove the `-d` flag if you want to track the model quantization progress in real-time. I don't recommend that you do since all of the activity is logged and you can also look at the .csv file to see the performance of the freshly quantized networks.

**NOTE**: If you don't want to run the entire CIFAR10 quantization script, I've included the quantized versions of the network in `quantized_models/experiment_2020-08-10_153501024022`.

## Model Plots and Figures

In the paper we show diagnostic plots for how the test accuracy behaves as each layer is quantized and the remaining layers are unquantized. We also show plots of how the test accuracy behaves as a function of the alphabet radius, and we histogram the weights for particular layers. All of the code for those plots are in `mlp_plots.py` and `cnn_plots.py`. I didn't intend for these files to be run as scripts, so you'll need to go in and manually change which models to look at and what model metrics .csv file to pull from.

## Minimal Working Example

Let's suppose I want to run the MNIST experiment. First, I'll train a network by calling the command 
```
docker run -dit --name train_container \
                -v /Users/elybrandadmin/Desktop/quantized_neural_networks/serialized_models:/serialized_models \
                -v /Users/elybrandadmin/Desktop/quantized_neural_networks/train_logs:/train_logs \
           quant_nnets python train_mnist_mlp.py
```
Once this script is done, it will serialize the MNIST network to `serialized_models/MNIST_Sequential2020-11-03_211410051651`. To quantize this network and cross-validate over the quantization parameters, I'll then execute the command
```
docker run -dit --name quant_container \
                -v /Users/elybrandadmin/Desktop/quantized_neural_networks/serialized_models:/serialized_models \
                -v /Users/elybrandadmin/Desktop/quantized_neural_networks/quantized_models:/quantized_models \
                -v /Users/elybrandadmin/Desktop/quantized_neural_networks/train_logs:/train_logs \
                -v /Users/elybrandadmin/Desktop/quantized_neural_networks/model_metrics:/model_metrics \
           quant_nnets python quantize_pretrained_mlp.py MNIST_Sequential2020-11-03_211410051651
```
As the script finishes quantizing the pre-trained analog network with a fixed bit-budget and alphabet radius, it logs the results of the quantized network's performance in `model_metrics/mnist_model_metrics_2020-11-03_211659124462.csv`, and then it serializes the quantized network into `quantized_models/`. That's it!

## ImageNet Network Compression

Full disclosure, I didn't bother trying to containerize the code for the ImageNet experiments. My hands were full at the time, and I was using a special fork of Tensorflow for my machine with a M1 chip. Please make any necessary modifications to the `requirements.txt` file if you wish to run these experiments in a container and you need a different version of tensorflow.

If you're an outsider to the computer vision community like I am, then you might be surprised at how much leg work you'll need to do to actually recreate any ImageNet results. You might also be wondering **which** ImageNet competition everyone is referring to. Let me spare you the trouble and redirect you to [Caleb Robinson's  GitHub repo](https://github.com/calebrob6/imagenet_validation) which was a godsend for me. Caleb's repo assumes you have the ILSVRC2012 validation set already. This data is surprisingly difficult to obtain, especially since the development kits are near impossible to find outside of Caleb's repo. I had to bit torrent the raw images. So much for being open sourced.

### Initial Preprocessing of Data

Once you have the ILSVRC2012 data, follow the instructions in Caleb's README file by putting the ILSVRC2012 data into a subdirectory titled `val` of Caleb's `data` directory. Move that new `data` directory into the parent directory of this GitHub repo. In other words, you should have the following file hierarchy:
```
. (quantized_neural_networks)
+-- data
|   +-- ILSVRC2012_validation_ground_truth.txt
|   +-- meta.mat
|   +-- synset_words.txt
|   +-- val
|   |   +-- ILSVRC2012_val_00000001.JPEG
|   |   +-- ...
|   |   +-- ILSVRC2012_val_00050000.JPEG
```

You're now ready to start pre-processing. I'll spare you the gory details and just tell you that all you need to do is run the script `python preprocess_imagenet.py` from within the `scripts` subdirectory. That basically downsamples all of the ImageNet data to a standard 224x224 grid. It then saves these preprocessed images into a new subdirectory `data/preprocessed_val` as `.npy` files. These preprocessing scripts were entirely written by Caleb Robinson. I just neatly packaged them into a python script and multiprocessed them so it runs quicker. On my machine, the preprocessing script takes roughly 3 minutes. 

### Quantizing an ImageNet Network

We're almost ready to start quantizing an ImageNet network. I can't tell you how much time I spent trying to choose from the [oodles of candidates in the Keras pretrained models](https://keras.io/api/applications/). I decided to keep it real simple and quantize VGG16 for two reasons. The first is that it is a massively overparameterized network, so our quantization algorithm can (and indeed does) really shine at low bit-budgets. The second reason is that it doesn't have any special layers other than pooling, Conv2D, Dense, and batch normalization. In other words, my quantization scripts worked out of the box.

You're welcome to try and quantize other pretrained models. I will let you know that the code should accomodate quantizing networks with `DepthwiseConv2D` layers (i.e. `MobileNet`) and networks with skip layers (i.e. `ResNet`). All you need to do to change which network you quantize is go into the `quantize_pretrained_imagenet.py` script, import the relevant network and its paired preprocessing function from Keras, and then  change `pretrained_model` and `preprocess_func` on lines 43 and 44, respectively. Note that the pretrained models from Keras come with preprocessing functions that are needed *in addition to* the preprocessing we did in the section before.

You may notice that I have the `is_quantize_conv2d` flag turned off. If you switch it to `True`, then the script will quantize convolutional layers. This will dramatically increase the time it takes to quantize networks like VGG16. And for marginal gain in memory savings: VGG16 has 90\% of its weights in the Dense layers, so I keep the flag turned off.

Now I have to come clean. In order to quantize ImageNet networks, you actually have to run a bash script. This is because there is a small but devastating memory leak that I just don't have time to hunt down. Maybe it's my fault, maybe it's the fault of running massive calculations for long periods of time and doing all sorts of I/O operations in the mean time. In any case, it only crashes the program if you try quantizing more than three or so networks in the same process. To avoid this, I've written a bash script in `scripts/imagenet_bash_script` which will iterate over a list of alphabet scalars by calling `python quantize_pretrained_imagenet.py [alphabet_scalar]` for the list of alphabet scalars listed in the bash script. If you want to try looking at different alphabet scalars, you need to modify the bash script.

Okay, so you're ready to start quantizing. At this point all you need to do is execute the bash script `$ ./imagenet_bash_script`.

Everything else about serializing the quantized models and so forth is the same as in the MNIST & CIFAR10 sections. The only difference is that since we're quantizing only one network per process, the model test accuracies are logged in `.csv` files with only one row. That's super annoying to compare to other hyperparameter configurations. So, I wrote yet another bash script to collect all files in the `model_metrics/` subdirectory which are generated from the ImageNet script and groups them into one `.csv` file. You're welcome.
      
# Appendix: Behind the Code
All of the work is done in `scripts/quantized_network.py`. Inside this file are two wrapper classes for Tensorflow networks. The `QuantizedNeuralNetwork` class handles networks with `Dense` layers. Any layer that is not a `Dense` layer is ignored. The `QuantizedCNN` class inherits the functionality of `QuantizedNeuralNetwork` and extends its functionality to handle quantizing `Conv2D, DepthwiseConv2D` layers. Any layers that are not `Dense, Conv2D` or `DepthwiseConv2D` are ignored. 

Both classes take as arguments a `Model` object (that is, an unquantized network), a `batch_size` which quantifies how many training samples to use when quantizing a given layer, an iterator `get_data` from which training samples are drawn to quantize the layers, a logging object `logger`, the number of bits `bits` to use in the quantization alphabet, and an alphabet radius `alphabet_scalar`. 

Layers are quantized successively. The only function that a user should call is `.quantize_network()`. This function makes calls to `._quantize_[dense_parallel, conv2D_parallel]_layer()`, which makes calls to `._quantize_[neuron_parallel, _quantize_channel_parallel_jit]()`. These latter two functions run the dynamical system proposed in the paper. The rest of the function calls in each class are pre-processing steps to shape the data into the appropriate dimensions. This is especially true for `QuantizedCNN` since each channel has to be quantized independently, and the data a channel filter acts on is a vectorized patch of an image. `._build_patch_array()` is responsible for getting these patches at every layer and then vectorizing them into a tensor across channels. 

These patch tensors can get unweildy very quickly, especially if lots of images are used to train a layer. To mitigate the memory pressure on RAM, these patch tensors are written to disk as a `hdf5` file. Once a layer is done quantizing, that `hdf5` file is removed from disk. All of these OS calls do make the network quantization run a bit slower, but these were necessary since I was running these experiments on constrained hardware.
