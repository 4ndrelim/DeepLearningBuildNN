# Build Neural Network with MNIST Database
This repository contains code for the construction of basic neural networks from scratch. Since it is a ground-up implementation, use of third-party libraries is kept to a minimum, with only *numpy* for efficient matrix computation and *matplotlib* to visualise the results, being used (at least up till Version 2 of the network). Dataset used is the MNIST digits database.

To better understand the code and computation in the algorithm, this [series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3Blue1Brown is highly recommended. Here he gives a comprehensive overview and provides intuition for the math behind the algorithm.

## Installation Guide & Usage
1. Clone with
```
git clone https://github.com/4ndrelim/DeepLearningBuildNN.git
```
2. Install third-party libraries (Note that this is sufficient up till Network V2.2; a requirements.txt will be made for Network 3). On the command line run:
```
$ pip3 install numpy
```
and
```
$ pip3 install matplotlib
```
3. Training your network 
    * To view just the numerical results (**only the finalized network of each version** is included here), run *test.py* in your preferred code editor or in the terminal:
    ```
    python3 test.py
    ```
    * To visualise the results of the different versions and sub-versions, run *visualise.py* or in the terminal:
    ```
    python3 visualise.py
    ```
    * comment-out unwanted training sections to save time
4. Toggling of hyperparameters
    * Networks in *visualise.py* and *test.py* have their hyperparams specified and toggled in *hyperparams.py*
    * It may be better to simply re-write hyperparams in each individual python file (in sub-directories under */visualisation*) if for some reasons, one wishes to run the several versions of Network 2 with different hyperparams in the same program (e.g *visualise.py*)
    * It should be unsurprising that the results of each epoch and final accuracy is the same over many iterations of the program, since the same pseudo-random seed was specified; comment-out the seed if this behaviour is unwanted
    * Hyperparams can still be better selected to further improve the model; most common and typical way of selection is done empirically as shown in [exploring suitable parameters](#version-2)
5. Saving & loading your network
    * A save and load function has been included in *network_2.py* to save the weights and biases of your trained model
    * Instructions on usage are documented with the function
    * Note that *network.py* does not have this save function but you should be able to copy-paste (save minor edits) since the implementation of save and load functions are independent of the network features

## Overview of Sample Results
With the hyperparameters specified in hyperparams.py:

Network V1.0: 95.12% success rate

<img src='./visualisation/network1/sample_accuracy.png' alt='Version 1.0' width='400'>
<br></br>

Network V2.2: 97.24% success rate

<img src='./visualisation/network2/observe_early_stopping/sample_early_stop.png' alt='Network 2.2' width='400'>
<br></br>

Netowrk V3.0: **PENDING**

Results can but emulated by running test.py and can likely be further be improved by training more epochs.

## Graphical Evaluation
The different graphs below can be re-produced by running visualise.py

### Version 1
Conventional implementation of a neural network. Read more [here](./visualisation/network1/README.md).

### Version 2
Here we explore and build a stronger network with the following modification:

a) L2 regularization

b) Cross-entropy function

c) Better initialization of weights

d) Early stopping

1. Version 2.0
    * Network is implemented with L2 regularization and uses cross-entropy cost function to improve performance. Read more [here](./visualisation/network2/view_learning2/README.md).

2. Version 2.1
    * Compare the effects of different initialization of weights. Read more [here](./visualisation/network2/weight_initialization_comparison/README.md).
    * Exploring suitable parameters (case study of learning rate) Read more [here](./visualisation/network2/test_learning_rate/README.md).

3. Version 2.2 
    * Network implemented with early stopping. Read more [here](./visualisation/network2/observe_early_stopping/README.md).

### Version 3
Network is implemented further with convolutional layers before dense layers.
*coming soon!*

## Resources & Credits
Referenced ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com).

**Note**: Some of the content in this repository was originally forked from ["neural-networks-and-deep-learning"](https://github.com/mnielsen/neural-networks-and-deep-learning) by Michael Nielson but I eventually decided to make a personal repository given vast personal changes.

Implementation of some parts may differ significantly from author's to ensure:

a) Further optimization

b) Additional features

c) Compatiblity with Python3

d) My own learning!

