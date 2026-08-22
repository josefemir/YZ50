#TODO: 3

import numpy as np

def loss_fn(predicted, true):

    #using formula loss = sum of all cost fn
    #cost fn = (predicted-expected)**2

    loss = np.sum((predicted - true)**2)
    return loss

###########################################

#TODO: 4

#some made up numbers
x=2
y_true = 10

#giving continous w values to plot correctly
w_list = np.linspace(-2,12,100)
loss_list = []

for w in w_list:
    pred = w*x
    loss = loss_fn(pred, y_true)
    #appending to list to plot each loss for every w value
    loss_list.append(loss)

import matplotlib.pyplot as plt

plt.plot(w_list, loss_list)
plt.xlabel("Parametre (w)")
plt.ylabel("Loss (Hata)")
plt.show()

###########################################

#TODO: 5

#this fn taking derivative using the formula, declaring h some small number to represent limit goes to 0
def gradient_calculator(w,x,y_true,h=0.0001):
    
    loss_1 = loss_fn(w*x,y_true)
    loss_2 = loss_fn((w+h)*x,y_true)

    gradient = (loss_2 - loss_1) / h
    return gradient

#AI ile tartışırken kodda Learning Rate olmasının gerekliliğinden bahsetti
#Videolarda Learning Rate kavramından bahsetmediği için kavramı Gemini'den öğrendim.
#Temelinde, türevin dikliğinin geçerli olduğu aralığı veriyor. Yani hesaplama sonucu gelen skaler olarak 56 büyüklükteki eğim sadece çok ufak birim için geçerli.
#Bu ufak birimi de lr = 0.01 olarak alabileceğimi öğrendim.

w_initial = -2.0
lr = 0.01

for i in range(50):
    #calculating the gradient and storing it in a variable in each iteration
    grad = gradient_calculator(w_initial,x,y_true)

    #optimizing the weight with respect to gradient value in that step
    w_initial = w_initial - grad*lr

print(f'Bulunan {w_initial}. Hedeflenen 5.0')