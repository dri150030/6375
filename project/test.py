import numpy as np
from layers import ConvolutionLayer,PoolingLayer,DenseLayer
#from scipy.signal import convolve2d,correlate2d
from cnn import CNN




X = np.random.randn(2,5,5)
#k = np.array([[[1,0],[0,0]],[[0,1],[0,0]],[[0,0],[-1,0]]])

C = ConvolutionLayer(3,2,X.shape)
Z = C.forward(X)


print(X)
#print(C.kernels)
print(Z)

#X2 = np.array([[[0,55,0,0],[20,0,41,33],[0,90,0,0],[0,57,0,95]]])

P = PoolingLayer(2)
Z = P.forward(Z)

#print(X)
print(Z)
