import numpy as np
from layers import ConvolutionLayer,PoolingLayer,DenseLayer
#from scipy.signal import convolve2d,correlate2d
from cnn import CNN


X = np.random.uniform(0,1,(1,28,28))
C = ConvolutionLayer(3,3,X.shape)
P = PoolingLayer(2)
D = DenseLayer(5,3 * 13**2)
N = CNN()

Z1 = C.forward(X)
Z2 = P.forward(Z1)
Z3 = D.forward(Z2)
Z4 = N.softmax(Z3)

#print(X.shape)
#print(Z1.shape)
#print(Z2.shape)
#print(Z3)
#print(Z4)

y = np.argmax(Z4)
dZ1 = N.backward(Z4,y)
dZ2 = D.backward(dZ1)
dZ3 = P.backward(dZ2)
dZ4 = C.backward(dZ3)

print(dZ1.shape)
print(dZ2.shape)
print(dZ3.shape)
print(dZ4.shape)


exit()


z = np.random.randn(5)**2

p = s.forward(z)
c = np.argmax(p)

dLdP = np.zeros(5)
dLdP[c] = -1 / p[c]


dLdZ = np.array([p if i != self.y_pred else p-1 for i,p in enumerate(self.P)])

print(z)
print(p)
print(dLdP)
print(dLdZ)


