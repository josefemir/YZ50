#TODO: 1

#writing our own Value class

import numpy as np
import math

class Value:
    #Backpointer olarak class içerisinde _children=() tanımlıyoruz. Burada amaç işlemin öncesinde hangi elemanların kullanıldığının takibini yapmak.
    #self._prev = set(_children) diyerek ._prev() kullanarak geçmişte kullanılan elemanları görebileceğiz. Aynı zamanda her işleme de gerekli taşımayı yapıyoruz


    #Bir diğer backpointer olarak class içerisinde _op='' tanımlıyoruz. Buradaki amacımız da Value object'in hangi işlemlere uğradığının takibi.
    #self._op = _op Her işleme gerekli taşımayı yapıyoruz.
    def __init__(self,data, _children=(), _op='', label =''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

        #gradient özellik olarak class içinde bulunmalı, initial olarak 0
        self.grad = 0
    
    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self,other):
        output = Value(self.data + other.data, (self,other),'+')
        return output

    def __mul__(self,other):
        output = Value(self.data * other.data, (self,other),'*')
        return output

    #Burada Neuron görevini tamamlamak için Value içerisinde tanh tanımladık (tanh = e**2x -1 / e**2x +1)
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) +1)
        o = Value(t, (self, ), 'tanh')
        return o


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
d = e+c; d.label='d'
f = Value(-2.0, label='f')

print(e._prev)
print(e.data)
print(d._prev)
print(d.data)

L =d*f; L.label='L'

print(L._prev)

#######################################

#TODO: 2

#manually calculating gradients
#Backprop da recursive olarak chain rule kuralının uygulanmasıdır (Karphaty)

#dL/dL=1
L.grad = 1.0

# dL/dd --> L = d*f --> dL/dd = f --> f=-2
d.grad = -2.0

# dL/df --> L = d*f --> dL/df = d --> d=4
f.grad = 4.0
 
# dL/dc --> dL/dd * dd/dc
# d = e+c --> dd/dc = 1
# dL/dd = -2.0
# dL / dc (CHAIN RULE) --> -2.0*1.0 = -2.0
c.grad = -2.0

# dL/de --> dL/dd * dd/de
# d = e+c --> dd/de = 1
# dL / de (CHAIN RULE) --> -2.0*1.0 = -2.0
e.grad = -2.0

# dL/da --> dL/dd * dd/de * de/da
# dL/da = -2*1*de/da
# de/da = b = -3
# dL/da = -6
a.grad = 6.0

# dL/db --> dL/dd * dd/de * de/db
# dL/db = -2*1*de/db
# de/db = a = 2
b.grad = -4.0

###Backprop through Neuron

#Burada Value class'ımıza tanh fonksiyonunu ekledik. Sigmoid, ReLU gibi bir düzeltme fonksiyonu. -1 ve +1 arasında değer almasını sağlıyor.

x1 = Value(2, label='x1')
x2 = Value(0, label='x2')

w1 = Value(-3, label='w1')
w2 = Value(1, label='w2')

b = Value(6.88113735870195432, label="b")

x1w1 = x1*w1
x2w2 = x2*w2

x1w1x2w2 = x1w1 + x2w2

# the argument of tanh func
n = x1w1x2w2 + b
o = n.tanh()

print(n)
print(o)

# o = tanh(n)
# do/do = 1
# do/dn = 1 - tanh(n)**2 = 1 - o.data**2

o.grad = 1
n.grad = 1 - o.data**2
print(n.grad)   #neredeyse 0.5 geliyor formülde yerine koyulduğunda

x1w1x2w2.grad = n.grad * 1.0
b.grad = n.grad * 1.0

# çünkü toplama işleminden türüyorlar. türevleri 1 geliyor, chain ruledaki çarpan 1.
x1w1.grad = n.grad * 1.0
x2w2.grad = n.grad * 1.0

# çarpma işleminden geliyorlar. türevleri çarpıldıkları eleman oluyor, chain rule çarpanları w1 ve x1.
x1.grad = n.grad * w1.data
w1.grad = n.grad * x1.data

print(x1.grad)
print(w1.grad)

x2.grad = n.grad * w2.data
w2.grad = n.grad * x2.data

print(x2.grad)
print(w2.grad)