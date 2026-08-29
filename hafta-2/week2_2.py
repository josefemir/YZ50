#TODO: 3

#.backward() metodu ile manuel değil otomatik gradyan hesaplama

import numpy as np
import math

class Value:
    def __init__(self,data, _children=(), _op='', label =''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

        self.grad = 0

        #backward özellik olarak class içerisinde bulunacak, initial olarak empty fn
        self._backward = lambda: None
    
    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self,other):
        output = Value(self.data + other.data, (self,other),'+')

        #burada neden fn içinde fn define ettiğini anlamadım.
        #eğer direkt olarak output._grad = self.grad *1 yazarsam o anda self.grad hesaplanmadığı için değerin 0 olacağını söyledi
        #burada fn define etmemizin sebebi closure kullanmak
        def _backward():

            # += kullanarak bir nörona bağlı 2 aynı değer olduğu zaman overwrite etmekten kurtuluyoruz.

            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
            #burada hiçbir şey return etmiyor, çünkü bu fn amacı değer döndürmek değil çağrıldığı zaman backprop yaparak bağlı node'ların grad güncellemek
        
        output._backward = _backward

        return output

    def __mul__(self,other):
        output = Value(self.data * other.data, (self,other),'*')

        #çarpma işleminde diğer değer katsayı olarak bir önceki grad ile çarpılır
        def _backward():

            # += kullanarak bir nörona bağlı 2 aynı değer olduğu zaman overwrite etmekten kurtuluyoruz.
            
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        
        output._backward = _backward
        #burada da if op='+' elif op='*' gibi işlemlere girmememizin sebebi geçmişinde hangi işlem varsa direkt onun içerisine girerek _backward getiriyor.
        #yani içeride bir çarpma işlemi kilitlenir. o._backward çağrıldığı an bu adrese gelip finalde elinde olan değerlerle bu çarpmayı yapar.

        return output

    #tanh türevi alınmış versiyonu 1-tanh**2 
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) +1)
        o = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2) * o.grad
            # local derivative * previous derivative --> Chain Rule
        
        o._backward = _backward
        return o

    #topolojik sıradan her node için backward çağıran actual backward tanımlıyoruz
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
    

x1 = Value(2, label='x1')
x2 = Value(0, label='x2')

w1 = Value(-3, label='w1')
w2 = Value(1, label='w2')

b = Value(6.88113735870195432, label="b")

x1w1 = x1*w1

x2w2 = x2*w2

x1w1x2w2 = x1w1 + x2w2

n = x1w1x2w2 + b
o = n.tanh()

"""
#manual calling process of ._backward() fn for all
o.grad = 1.0
o._backward()

print(n.grad)

n._backward()
print(x1w1x2w2.grad)

x1w1x2w2._backward()
print(x1w1.grad)
print(x2w2.grad)

x1w1._backward()
print(x1.grad)
print(w1.grad)

x2w2._backward()
print(x2.grad)
print(w2.grad)
"""

#aynı eleman birden fazla yazıldığında türev overwrite edilmesin, akümüle edilsin diye += yazımına geçtik
#zaten self.grad = 0 ile başlıyordu, bu sebeple += dediğimizde her seferinde üzerine 0 + new_grad şeklinde ilerleyecek
o.backward()

print(n.grad)
print(x1w1x2w2.grad)
print(x1w1.grad)
print(x2w2.grad)
print(x1.grad)
print(w1.grad)
print(x2.grad)
print(w2.grad)