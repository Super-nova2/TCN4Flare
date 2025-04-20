# TCN4Flare

## 1. Introduction
This is the official implementation of the TCN4Flare model.  
This model is based on the Temporal Convolutional Network (TCN) architecture and is specifically designed for the detection of flares in the light curves of AGNs.  

## 2. Model Architecture
The TCN4Flare model is a TCN model, using the keras-tcn (3.5.4) python package. The detailed architecture of the model can be seen in https://github.com/philipperemy/keras-tcn.

Some important arguments used in the TCN4Flare model are:
- input_shape: the shape of the input light curve. (None, None, 2)
- nb_filters: the number of filters in the convolutional layers. Fault is 128.
- kernel_size: the size of the convolutional kernel. Fault is 3.
- dilations: the list of the dilations used in the convolutional layers. Fault is [1, 2, 4, 8, 16, 32, 64, 128, 256].
- nb_stacks: the number of stacks of TCN layers. Falut is 1.
- return_sequences: False
- activation: the activation function used in the TCN layers. Fault is 'sigmoid'.
- dropout_rate: the dropout rate used in the TCN layers. Fault is 0.2.
- kernel_initializer: the kernel initializer used in the TCN layers. Fault is 'he_normal'.
- use_layer_norm=True.
- use_skip_connections=True.

## 3. Datasets
To detect flares in the light curves of AGNs, we need two kinds of data: LCs with and without flares.

### 3.1 LCs without flares
We use Zwiciky Transient Facility (ZTF) public data of AGNs as raw data. Firstly, we use traditional methods (**To be added**) to detect flares of these raw LCs and get LCs which are considered as no-flare by the flare-detection algorithm. Then we checked these LCs manually. Finally, we get a dataset of LCs without flares.

### 3.2 LCs with flares
