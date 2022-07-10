"""
Experimenting with different learning_rates to determine a suitable value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we plot the network's training cost against the number of epochs
as we vary the learning rate of each network.
"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../../../') # required if run_and_plot() were run in this script
import mnist_loader
import network_2
from hyperparams import NUM_EPOCHS, LEARNING_RATE, LAYERS, REG_PARAM, MINI_BATCH_SIZE, EARLY_STOPPING

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Customize Hyperparameters
# By default, uses the values specified in the root folder hyperparams.py
# Note you may wish to toggle display settings/scale
# on the y-axis should the results go out of range

##NUM_EPOCHS = 30
##LEARNING_RATE = 0.4
## HIDDEN_LAYER = [30]
##LAYERS = [784] + HIDDEN_LAYER + [10] # list of layers
##REG_PARAM = 5.0
##MINI_BATCH_SIZE = 10
####EARLY_STOPPING = 5

# To scale the accuracy to 100%
PLOT_SCALE_FACTOR = None # equals to size of validation data

# Manipulating pseudo-random state
# So you can get the exact same results as I did!
random.seed(123581321)
np.random.seed(123581321)

def run_and_plot(file_name, learning_rates):
    run_networks(file_name, learning_rates)
    make_plot(file_name, learning_rates)
    

def run_networks(file_name, learning_rates):
    """
    Train networks using three different values for the learning_rate,
    and store the cost curves in the 'multiple_lr.json'/given file, where
    it will be used later on to plot the curves
    """
    # References global variable
    global PLOT_SCALE_FACTOR

    # get and unzip data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)

    # get size of validation_data. This will be used to approrpiately scale plot
    PLOT_SCALE_FACTOR = len(validation_data)

    results = [] # stores results of each network
    for lr in learning_rates:
        print("\nTrain a network (other hyperparams are same as in network 2.1) using learning_rate = "+str(lr))
        network = network_2.Network(LAYERS, cost=network_2.CrossEntropyCost)
        results.append(
            network.SGD(training_data, NUM_EPOCHS, MINI_BATCH_SIZE, lr,
                        reg_param=REG_PARAM, eval_data = validation_data,
                        monitor_trng_cost=True))
        f = open(file_name, "w")
        json.dump(results, f)
        f.close()
        
def make_plot(file_name, learning_rates):
    f = open(file_name, "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    COLORS = COLORS = ['#FFA933', '#2A6EA6', '#FF492A']
    for lr, result, color in zip(learning_rates, results, COLORS):
        _, _, trng_cost, _ = result
        # points plotted with connected small circles as specified by "o-"
        ax.plot(np.arange(0, len(trng_cost), 1), trng_cost, "o-",
                label="lr = "+str(lr),
                color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show(block=False)
        
    
    
        
    
    
    
