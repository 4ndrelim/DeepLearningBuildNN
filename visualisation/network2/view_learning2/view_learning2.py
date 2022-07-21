"""
View accuracy on validation dataset using network 2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we plot an improved network's performance in its
classification of digits over a specified number of epochs

Improvements:
1. L2 Regularisation
2. Cross-entropy cost function

"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../../../') # required if run_and_plot_network() were run in this script
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

                       
def run_and_plot_network(filename):
    """
    Train the network using randomized unscaled
    (tend to be large) starting weights.
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
    print("~Training Network 2.0~")
    network.old_initializer()
    vc, va, tc, ta \
        = network.SGD(training_data, NUM_EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, reg_param=REG_PARAM,
                  eval_data=validation_data, 
                  monitor_eval_acc=True)

    print(f"\nAccuracy on test set: {100.0 * network.accuracy(test_data)/len(test_data)}%")
    
    f = open(filename, "w")
    json.dump({"initialization": [vc, va, tc, ta]}, f)
    f.close()
    
    make_plot(filename)

def make_plot(filename):
    """
    Load the results from the file ``filename``, and generate the
    corresponding plot.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    _, va, _, _ = data["initialization"]
   
    # Convert raw classification numbers to percentages, for plotting
    validation_results = [100.0 * x/PLOT_SCALE_FACTOR for x in va]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, NUM_EPOCHS+1, 1), validation_results, color='#FFA933',
            label="Accuracy (%) after each epoch")
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title('Classification accuracy')
    plt.legend(loc="lower right")
    fig.canvas.set_window_title('network2_accuracy')
    plt.show(block=False)
