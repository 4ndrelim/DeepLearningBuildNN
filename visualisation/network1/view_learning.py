"""
View accuracy on validation dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we plot the network's improvement in its classification
of digits over a specified number of epochs

"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../../') # required if run_and_plot_network() were run in this script
import mnist_loader
import network
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
##MINI_BATCH_SIZE = 10

# To scale the accuracy to 100%
PLOT_SCALE_FACTOR = None # equals to size of validation data

                       
def run_and_plot_network(filename):
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

    net = network.Network(LAYERS)
    print("~Training Network 1~")
    validation_results = net.SGD(training_data, NUM_EPOCHS, MINI_BATCH_SIZE,
                                    LEARNING_RATE, validation_data=validation_data)

    print(f"\nAccuracy on test set: {100 * net.evaluate(test_data)/len(test_data)}%")
    
    f = open(filename, "w")
    json.dump({"accuracies": validation_results}, f)
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
    validation_results = data["accuracies"]
   
    # Convert raw classification numbers to percentages, for plotting
    validation_results = [100.0 * x/PLOT_SCALE_FACTOR for x in validation_results]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, NUM_EPOCHS, 1), validation_results, color='#FFA933',
            label="Accuracy (%) after each epoch")
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title('Classification accuracy')
    plt.legend(loc="lower right")
    fig.canvas.set_window_title('network1_accuracy')
    plt.show(block=False)

