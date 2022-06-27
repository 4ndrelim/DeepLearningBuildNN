"""
A program to test the constructed neural networks
"""

import mnist_loader
import network
import network_2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# print(list(training_data)[5][0].shape) numpy array! 784 input layer


"""
Neural netowrk 1.0
SGD + backprop using sigmoid function
"""
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.13, test_data = test_data)


"""
Neural network 2.0
Similar to 1.0, with the addition of
a) L2 regularization
b) cross-entropy function
c) better initialization of weights
"""
net = network_2.Network([784, 30, 10], cost=network_2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.4,
        reg_param = 5.0,
        eval_data = validation_data,
        monitor_eval_cost=False,
        monitor_eval_acc=True,
        monitor_trng_cost=False,
        monitor_trng_acc=False)




