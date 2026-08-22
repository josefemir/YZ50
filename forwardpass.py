#TODO: 1

W0  = -1.5
W1  = 3
W2  = 0.4

a0 = 0.5
a1  = 0.1
a2  = 0.7

b0  = 12

import math

#forwardpass using Sigmoid

def sigmoid(x):
    return 1/(1+(math.exp(-x)))

def forwardpass_sigmoid():
    return  sigmoid((W0*a0)+(W1*a1)+(W2*a2)+b0)


#forwardpass using ReLU

def ReLU(x):
    if x >= 0:
        return x
    else:
        return 0

def forwardpass_relu():
    return ReLU((W0*a0)+(W1*a1)+(W2*a2)+b0)