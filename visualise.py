import sys

"""
A program to visualise the effects of certain features/results of the neural network

NOTE:
1. GRAPHS WILL BE CONCURRENTLY DISPLAYED AT THE END OF ALL THE EXECUTON. COMMENT OUT PARTS YOU DO NOT WISH TO OBSERVE TO SAVE TIME.
2. Default hyperparameters are specified in the hyperparams.py file in the root folder.
3. If you are running more than 1 network and wish to customize hyperparams for each network, uncomment the code that handles re-writing
   hyperparameters in the respective program files in visualisation/ folder.
"""


FILENAME = "my_model_results"


"""
Network 1.0 Observe Improvement of Network
"""
sys.path.append("./visualisation/network1/")
import view_learning as vl
path = "./visualisation/network1/"
path = path + FILENAME
vl.run_and_plot_network(path)
print("Note: Accuracy results of trained model is saved at " + path[2:] + "\n\n")


"""
Network 2.0 Improvement of Network with L2 regularization & cross entropy cost
"""
sys.path.append("./visualisation/network2/view_learning2/")
import view_learning2 as vl2
path = "./visualisation/network2/view_learning2/"
path = path + FILENAME
vl2.run_and_plot_network(path)
print("Note: Accuracy results of trained model is saved at " + path[2:] + "\n\n")


"""
Network 2.1 Effects of Weight Initialization
"""
sys.path.append("./visualisation/network2/weight_initialization_comparison/")
import compare_initialization as cmp_init
path = "./visualisation/network2/weight_initialization_comparison/"
path = path + FILENAME
cmp_init.compare(path)
print("Note: Accuracy results of trained model is saved at " + path[2:] + "\n\n")


"""
Network 2.1 Experimenting with different learning rates
"""
sys.path.append("./visualisation/network2/test_learning_rate/")
import test_lr
path = "./visualisation/network2/test_learning_rate/"
path = path + FILENAME
LEARNING_RATES = [0.05, 0.5, 2.5]
test_lr.run_and_plot(path, LEARNING_RATES)
print("Note: Accuracy results of trained models are saved at " + path[2:] + "\n\n")



"""
Network 2.2 Use of early stopping to identify suitable number of epochs
"""
sys.path.append("./visualisation/network2/observe_early_stopping")
import early_stop
path = "./visualisation/network2/observe_early_stopping/"
path = path + FILENAME
early_stop.plot_network_with_early_stopping(path)
print("Note: Accuracy results of trained model is saved at " + path[2:] + "\n\n")
