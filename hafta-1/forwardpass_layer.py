#TODO: 2

#input has 3 neurons, layer has 2 neurons

#activation for input neurons
a0 = 0.5
a1  = 0.1
a2  = 0.7

#weight of every neuron respect to neuron no-0
w00 = -3
w01 = -2
w02 = 5

#weight of every neuron respect to neuron no-1
w10 = 0.4
w11 = 4
w12 = -3

#biases for each
b0 = 0.4
b1 = 0.2

import math

def sigmoid(x):
    return 1/(1+(math.exp(-x)))

def forwardpass_sigmoid():
    #sigmoid inputs for each neurons (using formula W*a+b)
    z0 = w00*a0 + w01*a1 + w02*a2 + b0
    z1 = w10*a0 + w11*a1 + w12*a2 + b1

    out0 = sigmoid(z0)
    out1 = sigmoid(z1)

    return [out0,out1]
