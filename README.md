# Build neural network with MNIST database
This repository contains code for the construction of basic neural networks from scratch. Dataset used is the MNIST digits database.

## Installation Guide

## Overview
With the hyperparameters specified in hyperparams.py:

Network V1: 95.12% success rate

Network V2.0: 97.24% success rate

Results can but emulated by running test.py

## Graphical evaluation
### Version 1
Conventional implementation of a neural network with sigmoid function and quadratic cost function:

![V1 Classification accuracy](./visualisation/network1/sample_accuracy.png)

### Version 2
1. Observe the impact of scaling the initialized weights by 1/sqrt(n) where n is the number of input weights:

    * ![Initialized weights comparison](./visualisation/network2/weight_initialization_comparison/sample_weightInitCompare.png)

    * Notice that while both eventually converge to the same accuracy results, it appears as though the graph using the older approach is trailing that of the new graph. This technique of initializing weights become more pronounced and significant when training even larger datasets with high expected number of training epochs required (
    [network prefers learning small weights](https://datascience.stackexchange.com/questions/29682/understanding-regularisation-and-a-preference-for-small-weights)).

2. Network2 implemented with early stopping
    * ![Network 2 with early stopping](./visualisation/network2/observe_early_stopping/sample_early_stop.png)




## Resources
Referenced ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com).
Implementation of some parts may differ from author's to ensure:

a) Further optimization

b) Additional features

c) Compatiblity with Python3

d) My own learning!

