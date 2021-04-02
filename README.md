# Quantized Neural Networks

This repository contains code for the experiments in the manuscript "[A Greedy Algorithm for Quantizing Neural Networks](https://arxiv.org/abs/2010.15979)" by Eric Lybrand and Rayan Saab (2020). These experiments include training and quantizing two networks: a multilayer perceptron to classify MNIST digits, and a convolutional neural network to classify CIFAR10 images. 

## Setting Up Docker
To run the code in a docker container, you'll first need to download [Docker](https://www.docker.com/get-started). Once you've got Docker installed, navigate to the github repo on your machine and run the following CLI command: `docker build --tag quant_nnets .` (don't forget the period). This will build a docker image with the python 3.7.10 base image and download other required python packages from `requirements.txt`. This is a somewhat large docker image since we have to download Tensorflow, but it's significantly smaller than the bloated AWS DLAMI. It may take a minute or two to build. To see that the docker image has built successefully, run the command `docker images` and look for `quant_nnets` under the repository column.

Another thing you'll have to do is to modify how much memory Docker Desktop is allotted. In Docker Destop, navigate to "Resources", then to "Advanced". I was able to run the scripts with 16GB memory allottment (half of my machine's RAM). This may be overkill, but I didn't bother to fine tune. Also, for the CIFAR10 quantization script the disk pressure will get up to ~30GB, so make sure you have that much space on disk for this experiment.

## Running Experiments
Once the image is built, you can start running the experiments. These experiments are set up so that model training and model compression occur in two separate scripts. Once we have a trained network, that network is saved in the directory `serialized_models`. To persist that trained model on your local machine, we'll use Docker volumes.

### Network Training

To train a network on a docker container, run the following command
```
docker run -dit --name train_container \
                -v [absolute/path/to/repo]/serialized_models:/serialized_models \
                -v [absolute/path/to/repo]/train_logs:/train_logs \
           quant_nnets python [train_mnist_mlp.py, train_cifar10_cnn.py]
```

Importantly, to use Docker volumes you need to specify the absolute path to the github repo and place that where you see `[absolute/path/to/repo]` in the above command. You also should choose either `train_mnist_mlp.py` or `train_cifar10_cnn.py`  to run. Training the MNIST network should take only a few minutes, whereas the CIFAR10 network will take significantly longer. On my machine which didn't use any GPUS for training it took roughly 16 hours. 

**NOTE**: The `-d` flag will run the docker container in the background, so you won't see any output onto the command line during model training. If you want to track the model training progress in real-time, you can replace the `-dit` flag with `-it`. This runs the docker container in attached mode. You will see the verbose output from the Keras training in your terminal. 

**NOTE**: The analog CIFAR10 network we used in our experiments is included in `serialized_models` if you don't want to train your own network. 

### Network Compression
Once you have a pretrained network saved in `serialized_models` you're ready to run the network quantization script. Both model quantization scripts `quantize_pretrained_mlp.py` and `quantize_pretrained_cnn.py` define parameter grids at the top of their respective python files which they cross-validate over. Those parameter grids include:
1. choices for the number of training data to be used to learn the quantization,
2. the number of bits to use in the quantization alphabet, defined as `bits`,
3. the radius of the quantization alphabet, defined as `alphabet_scalars`.

In the paper, we only validate over the last two parameters. You're welcome to play around with the number of samples used to train the quantization. Just note that if you do make any modifications to the scripts, you'll need to rebuild the Docker container by again calling docker `build --tag quant_nnets .` Fortunately, Docker builds containers incrementally so you won't have to wait as long as you did the first time since all of the requirements will have already been downloaded.

**BEWARE** that training the quantization cannot be done with mini-batches of data! You will find that using the entire CIFAR10 dataset to train the quantization requires an enormous amount of disk space (think: for a convolutional layer with 256 filters, every image gets transformed from having 3 channels to 256 channels; further, instead of acting on the entire image the network acts on patches of data. That means the number of training images is not equal to the number of "samples" to learn the quantization; the number of samples is actually magnified by the number of patches and the number of channels). You'll also happen to find, like I did when I tried using the entire training data set, that you don't get that much better performance than using a much smaller subset on the order of 1000 images. I've set it up so that the training data for the quantization at each layer is saved to disk and is read one at a time to prevent running out of RAM if your local machine is constrained with RAM or if you really want to go overboard with the number of images.

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

**NOTE**: Running the quantization script for both MNIST and CIFAR10 networks takes a while because of the number of parameters to cross-validate over. For the MNIST data set, we also use half the training set (25000 images) which is far more than is needed to get a competitive quantization, and the layers are fairly wide. Prior to multiprocessing, it took on the order of 30 minutes to quantize a MNIST network. With multiprocessing, it only takes 10 minutes to quantize a MNIST network on my ancient late 2013 Macbook Pro with 4 cores. The CIFAR10 network takes some time because the network is deep-ish, and the code requires OS calls to save the output of previous layers to disk so we don't run out of RAM. Prior to multiprocessing, one round of quantization with a fixed parameter configuration and using 1000 images took about 30 minutes for CIFAR10. With multiprocessing, it now takes 20 minutes. 

**NOTE (for the insanely curious)**: If you're disappointed like I am about the meager speed up from multiprocessing for quantizing the CIFAR10 network, I can tell you what the bottlenecks are. The first bottleneck is that when we go to quantize a convolutional layer I end up "vectorizing" the data by chunking up each channel into patches and vectorizing those patches. I then save these giant patch tensors to disk. All of this is handled in `_segment_data2D()`. Inside `_segment_data2D()`, I end up calling a Tensorflow function `extract_patches()`. For some bizarre reason, I can multiprocess this routine just fine if I only feed in one training image. And yet, if I feed in more than one training image I get deadlock. I traced the deadlock to `extract_patches()`. [It turns out that this function is actually calls an analogous function from the Eigen library](https://stackoverflow.com/questions/46858070/implementation-of-tf-extract-image-patches), which is written in C++. I did not bother to try and read any further as to why I would be getting deadlock here, so for now multiprocessing is not enabled for generating these patch tensors. I guess I could write my own `extract_patches()` function, but that's a problem for future Eric.

The second bottleneck is simply due to the fact that I did not end up using that many training data to learn the quantized networks. Since the filters are so small for CIFAR10 (3x3, to be exact), it doesn't take very long to quantize a single filter. In other words, the time it takes to serialize the relevant data to send from the main CPU to the other CPUs is roughly on the order of the time it takes to quantize a filter. I speculate that if we were to use lots of training data to learn the quantized (which appears to be unnecessary to get competitive performance), then quantizing filters would take longer and multiprocessing would buy you more of a speedup.

**NOTE**: As before, remove the `-d` flag if you want to track the model quantization progress in real-time. I don't recommend that you do since all of the activity is logged and you can also look at the .csv file to see the performance of the freshly quantized networks.

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

# Appendix: Behind the Code
All of the work is done in `scripts/quantized_network.py`. Inside this file are two wrapper classes for Tensorflow networks. The `QuantizedNeuralNetwork` class handles networks with `Dense` layers. Any layer that is not a `Dense` layer is ignored. The `QuantizedCNN` class inherits the functionality of `QuantizedNeuralNetwork` and extends its functionality to handle quantizing `Conv2D` layers. Any layers that are not `Dense` or `Conv2D` are ignored. 

Both classes take as arguments a `Model` object (that is, an unquantized network), a `batch_size` which quantifies how many training samples to use when quantizing a given layer, an iterator `get_data` from which training samples are drawn to quantize the layers, a logging object `logger`, the number of bits `bits` to use in the quantization alphabet, and an alphabet radius `alphabet_scalar`. 

Layers are quantized successively. The only function that a user should call is `.quantize_network()`. This function makes calls to `._quantize_[dense_parallel, conv2D_parallel]_layer()`, which makes calls to `._quantize_[neuron_parallel, filter2D_parallel]()`. These latter two functions run the dynamical system proposed in the paper. The rest of the function calls in each class are pre-processing steps to shape the data into the appropriate dimensions. This is especially true for `QuantizedCNN` since each channel has to be quantized independently, and the data a channel filter acts on is a vectorized patch of an image. `._get_patch_tensor[_parallel]()` is responsible for getting these patches at every layer and then vectorizing them into a tensor across channels. 

These patch tensors can get unweildy very quickly, especially if lots of images are used to train a layer. To mitigate the memory pressure on RAM, these patch tensors are written to disk as a `hdf5` file. Once a layer is done quantizing, that `hdf5` file is removed from disk. All of these OS calls do make the network quantization run a bit slower, but these were necessary since I was running these experiments on constrained hardware.
