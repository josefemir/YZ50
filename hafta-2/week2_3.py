#TODO: 4

#breaking down tanh: exp, divide, power

import numpy as np
import math

class Value:
    def __init__(self,data, _children=(), _op='', label =''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

        self.grad = 0

        self._backward = lambda: None
    
    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self,other):
        #direkt integer ekleyebilmek için other eğer bir Value class object değilse yapan if döngüsü
        other = other if isinstance(other,Value) else Value(other)

        output = Value(self.data + other.data, (self,other),'+')


        def _backward():

            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        
        output._backward = _backward

        return output

    # burada karphaty ele almamış olsa da 2+a durumunda da patlamamak için __radd__ ekledim. (ai tavsiyesi)
    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * (-1)

    def __sub__(self,other):
        return self + (-other)

    def __rsub__(self,other):
        return self + (-other)

    def __mul__(self,other):
        #direkt integer çarpabilemk için other eğer bir Value class object değilse yapan if döngüsü
        other = other if isinstance(other,Value) else Value(other)

        output = Value(self.data * other.data, (self,other),'*')

        def _backward():

            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        
        output._backward = _backward

        return output

    def __pow__(self,other):
        assert isinstance(other, (int,float))
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            # üslü ifadenin türevi değerin başa gelip üssün 1 azalması (dx^2/dx = 2x)
            # burada neden other.data değil other kullandığımızı anlamadım
            # baştaki assert ile giren other değerini integer veya float yaptığımız için .data demeye gerek kalmıyor
            # diğer metodlarda Value olarak oluşturduğumuzdan .data ile çağırmaya devam ediyoruz

            self.grad += out.grad * (other * self.data**(other - 1))
        out._backward = _backward   

        return out

    #fallback scenario multiplication
    def __rmul__(self, other):
        return self * other

    # sıfırdan division kuralı tanımlayıp _backward tanımlamak yerine bölmeyi çarpma cinsinden tanımlayarak kodu modüler tutmak amacıyla 1/n ile çarpıyoruz.
    def __truediv__(self,other):
        return self * (other**-1)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) +1)
        o = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2) * o.grad
        
        o._backward = _backward
        return o

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            # d(e^x)/dx = e^x --> d(e^2x)/dx = 2*e^x
            # local derivative * previous derivative --> Chain Rule
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

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
    
a = Value(32)
print(a+2)
print(2*a)

print(a.exp())

a = Value(4.0)
b = Value(3.0)
print(a/b)
print(a-b)


x1 = Value(2, label='x1')
x2 = Value(0, label='x2')

w1 = Value(-3, label='w1')
w2 = Value(1, label='w2')

b = Value(6.88113735870195432, label="b")

x1w1 = x1*w1

x2w2 = x2*w2

x1w1x2w2 = x1w1 + x2w2

n = x1w1x2w2 + b

# 1st approach
o = n.tanh()
print(o)

# 2nd approach --> implementing the tanh formula by using e
e = (2*n).exp()
o = (e-1) / (e+1)
print(o)

#3rd approach --> implementing by PyTorch

import torch

x1 = torch.Tensor([2.0]).double() ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.88113735870195432]).double(); b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)
print(o.data)
print(o.data.item())

o.backward()

######################################

#TODO: 5

#building neuron, layer, MLP classes

import random

class Neuron():
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,x):
        # Neuron tanımlayıp içerisine bir değer soktuğumuzda sayısal olarak hangi değeri döndüreceğini tanımlarız.
        # zip metodu sıra sıra her elemanı eşleştirmeye yarar, biz de her bir weight ile her bir value'yu eşleştirip çarpacağımız için zip kullanacağız

        act = sum((wi * xi for wi,xi in zip(self.w,x)), self.b)
        # her bir w ve x değerini çarptı, ardından bütün çarpımları topladı
        # ardından activation formülü --> w*x+b --> ", self.b" toplamın hangi değerden başlayacağını tanımlar

        out = act.tanh()
        return out

    def parameters(self):
        #bütün parametreleri tek seferde çağırmak için ekledi
        #n.w[0].data, n.w[1].data gibi yazmak yerine bununla hepsini çağırabiliyoruz
        return self.w + [self.b]
    

class Layer():
    def __init__(self,nin,nout):
        # bütün nöronları bir liste olarak döndürür
        # nout --> how many neurons do you want in your layer
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self,x):
        # n in self.neurons --> yukarıdaki listedeki her bir nöronun içerisine x değişkenini atayarak işlem yaptırır
        # sonucunda bütün işlemleri outs'a yazar, bir liste olarak döndürür
        outs = [n(x) for n in self.neurons]

        #son katmanda tek eleman varsa liste içerisinde yazdırmasındansa direkt numerik değeri basması için
        return outs[0] if len(outs) == 1 else outs

    def parametres(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params

class MLP():
    def __init__(self,nin,nouts):
        # burada nin (ilk) girdisiyle nouts listesini toplayarak input layer dahil bütün layerları sz içinde store eder
        sz = [nin] + nouts
        
        #sz listesindeki her bir eleman için layerları sıra sıra oluşturur i=0,i=1,...
        # Layer(sz[i],sz[i+1]) ifadesinde Layer __init__ tanımından hareketle sz[i]=nin / sz[i+1] =nout olur
        # sz[i] --> listedeki i numaralı eleman (sz = nin+[nouts])
        # buradaki i ve i+1 döngüsü ve range(len(nouts)) sayesinde son layera kadar her layerı oluşturup layers içerisinde store eder
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            # burada her bir layerda bir önceki adımdan gelecek değerleri yeniden tanımlarız
            # aşağıdaki örnek kodda ilk adımda x = [2.0,3.0,-1.0] olan 3 değerli bir liste
            # layer(x) diyerek elde ettiği sonuç 4 değerli bir liste oluşturur, sonrasında 4 değerli farklı bir liste olarak x kendini günceller
            x = layer(x)
        return x

    def parameters(self):
        params=[]
        for layer in self.layers:
            params.extend(layer.parametres())

        return params
        
x = [2.0,3.0]
n = Neuron(2) # 2 dimensional neuron
print(n(x))

n = Layer(2,3) # 2 dimensional 3 neurons
print(n(x))

x = [2.0,3.0,-1.0]
n= MLP(3,[4,4,1]) #nin = 3 / nouts = [4,4,1]
print(n(x))

"""
mlp(x)
  │
  └──► layer(x)                   (MLP, Layer'ı çağırır)
         │
         ├──► n1(x) ──► 0.71      (Layer, 1. nöronu çağırır -> sayı döner)
         ├──► n2(x) ──► -0.45     (Layer, 2. nöronu çağırır -> sayı döner)
         ├──► n3(x) ──► 0.12      (Layer, 3. nöronu çağırır -> sayı döner)
         └──► n4(x) ──► 0.89      (Layer, 4. nöronu çağırır -> sayı döner)
         │
         └──► outs = [0.71, -0.45, 0.12, 0.89]  (Layer 4 sayıyı MLP'ye teslim eder)
  │
  x artık [0.71, -0.45, 0.12, 0.89] oldu.
  Döngü 2. katmana geçer...
  """

xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
  ]

ys = [1.0,-1.0,-1.0,1.0]

for k in range(110):
    # forward pass
    ypred = [n(x) for x in xs]

    #error calculation
    loss = sum((ygt-yout)**2 for ygt, yout in zip(ys, ypred))

    #backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    #gradient descent
    for p in n.parameters():
        # gradyanın tam tersi yönünde çok ufak adım atarak gradyanı minimize etmek amacındayız
        p.data += -0.05 * p.grad
    
    print(k, loss.data)
