import sys


print("Note: Graphs will be concurrently displayed at the end of all the below execution.\nYou should only uncomment parts you'd like to observe.")
FILENAME = "my_model_results"


"""
Network 1.0 Observe Improvement of Network
"""
sys.path.append("./visualisation/network1/")
import view_learning as vl
path = "./visualisation/network1/"
path = path + FILENAME
vl.run_and_plot_network(path)
print("Note: Accuracy results of trained model is saved at visualisation/network1/\n\n")


"""
Network 2.0 Effects of Weight Initialization
"""
sys.path.append("./visualisation/network2/")
import compare_initialization as cmp_init
path = "./visualisation/network2/"
path = path + FILENAME
cmp_init.compare(path)
print("Note: Accuracy results of trained model is saved at visualisation/network2/\n\n")
