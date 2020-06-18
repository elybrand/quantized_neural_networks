import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib osx

df = pd.read_csv('../model_metrics/mnist_model_metrics_2020-06-15_16:51:23.839830.csv')

# Select the parameter you wish to track.
grp = df.groupby(by='kernel_size')
grp_keys = list(grp.groups.keys())

# Plot the average test accuracy for each kernel size group.
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.plot(grp_keys, grp.mean()['analog_test_acc'], '-o')
ax.plot(grp_keys, grp.mean()['sd_test_acc'], '-o')
ax.plot(grp_keys, grp.mean()['msq_test_acc'], '-o')
ax.set_xlabel('Kernel Size', fontsize=18)
ax.set_ylabel('Test Accuracy', fontsize=18)
ax.set_ylim([0.5,1])
ax.set_xticks(grp_keys)
ax.set_yticks(np.arange(0.5,1.01,0.1))
ax.set_title('MNIST Test Accuracy vs Kernel Size', fontsize=22)
ax.legend(['Analog Network', r'$\Sigma\Delta$ Net', 'MSQ Net'], fontsize=14)

caption = f"Caption: Convolutional Neural Network with 2 convolutional layers and one dense layer trained on MNIST."\
f" The number of filters for each layer are 64 and 32, respectively. Batch normalization layers were added after each Conv2D layer. We used ReLu activations"\
f" and initialized weights with the GlorotUniform distribution. Strides were kept fixed at 2x2. "\
f" The above graph is given after averaging over all parameters and trials while keeping kernel size fixed. Each value therefore has {grp.size().values[0]}"\
" instances it averages over."
fig.text(0.5, 0.01, caption, ha='center', wrap=True, fontsize=12)