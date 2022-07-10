# Build Neural Network with MNIST Database
This repository contains code for the construction of basic neural networks from scratch. Dataset used is the MNIST digits database.

## Installation Guide

## Overview
With the hyperparameters specified in hyperparams.py:

Network V1.0: 95.12% success rate

<img src='./visualisation/network1/sample_accuracy.png' alt='Version 1.0' width='200'>
<br></br>

Network V2.2: 97.24% success rate

<img src='./visualisation/network2/observe_early_stopping/sample_early_stop.png' alt='Network 2.2' width='200'>
<br></br>

Results can but emulated by running test.py

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

## Resources & Credits
Referenced ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com).

**Note**: Some of the content in this repository was originally forked from ["neural-networks-and-deep-learning"](https://github.com/mnielsen/neural-networks-and-deep-learning) by Michael Nielson but I eventually decided to make a personal repository given vast personal changes.

Implementation of some parts may differ significantly from author's to ensure:

a) Further optimization

b) Additional features

c) Compatiblity with Python3

d) My own learning!

