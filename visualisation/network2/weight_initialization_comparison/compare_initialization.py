"""
Weight Initialization Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This program shows how weight initialization affects training.
In particular, we'll plot out how the classification accuracies improve using:

a) large/random starting weights (the one used in network 1.0),
   whose standard deviation is 1,
   
b) the new default starting weights, whose standard deviation is 1 over the
   square root of the number of input neurons.

"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../../../') # required if compare() was run in this script
import mnist_loader
import network_2
from hyperparams import NUM_EPOCHS, LEARNING_RATE, LAYERS, REG_PARAM, MINI_BATCH_SIZE

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Customize Hyperparameters
# By default, uses the values specified in the root folder hyperparams.py
# Note you may wish to toggle display settings/scale
# on the y-axis should the results go out of range
##NUM_EPOCHS = 30
##LEARNING_RATE = 0.4
##HIDDEN_LAYER = [30]
##LAYERS = [784] + HIDDEN_LAYER + [10] # list of layers
##REG_PARAM = 5.0
##MINI_BATCH_SIZE = 10

# To scale the accuracy to 100%
PLOT_SCALE_FACTOR = None # equals to size of validation data


def compare(filename):
    run_network(filename)
    make_plot(filename)
                       
def run_network(filename):
    """
    Train the network using both the default and the
    large starting weights.
    Store the results in the file with name ``filename``,
    where they can later be used by ``make_plots``.
    """
    # References global variable
    global PLOT_SCALE_FACTOR
    
    # Manipulating pseudo-random state
    # So you can get the exact same results as I did!
    random.seed(123581321)
    np.random.seed(123581321)

    # get and unzip data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)

    # get size of validation_data. This will be used to approrpiately scale plot
    PLOT_SCALE_FACTOR = len(validation_data)

    network = network_2.Network(LAYERS, cost=network_2.CrossEntropyCost)
    print("~Train Network 2.1 with cross-entropy cost fn and L2 regularization using the default (scaled) starting weights~")
    default_vc, default_va, default_tc, default_ta \
        = network.SGD(training_data, NUM_EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, reg_param=REG_PARAM,
                  eval_data=validation_data, 
                  monitor_eval_acc=True)
    print(f"\nAccuracy on test set with default weights initialization: {100.0 * network.accuracy(test_data)/len(test_data)}%")

    # zip object has been unzipped and used so re-initialize
    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("\n~Train Network 2 with cross-entropy cost fn and L2 regularization using the large (unscaled)/old starting weights~")
    network.old_initializer()
    large_vc, large_va, large_tc, large_ta \
        = network.SGD(training_data, NUM_EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, reg_param=REG_PARAM,
                  eval_data=validation_data, 
                  monitor_eval_acc=True)
    print(f"\nAccuracy on test set with large weights initialization: {100.0 * network.accuracy(test_data)/len(test_data)}%")
    
    f = open(filename, "w")
    json.dump({"default_initialization":
               [default_vc, default_va, default_tc, default_ta],
               "old_initialization":
               [large_vc, large_va, large_tc, large_ta]}, 
              f)
    f.close()

def make_plot(filename):
    """
    Load the results from the file ``filename``, and generate the
    corresponding plot.
    """
    f = open(filename, "r")
    results = json.load(f)
    f.close()
    default_vc, default_va, default_tc, default_ta = results[
        "default_initialization"]
    large_vc, large_va, large_tc, large_ta = results[
        "old_initialization"]
    # Convert raw classification numbers to percentages, for plotting
    default_va = [100.0 * x/PLOT_SCALE_FACTOR for x in default_va]
    large_va = [100.0 * x/PLOT_SCALE_FACTOR for x in large_va]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, NUM_EPOCHS+1, 1), large_va, color='#FFA933',
            label="Old approach to weight initialization")
    ax.plot(np.arange(0, NUM_EPOCHS+1, 1), default_va, color='#2A6EA6', 
            label="New approach to weight initialization")
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title('Classification accuracy')
    plt.legend(loc="lower right")
    fig.canvas.set_window_title('network2_weightInitCompare')
    plt.show(block=False)

