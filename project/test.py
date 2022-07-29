import numpy as np
from layers import ConvolutionLayer,PoolingLayer,DenseLayer
#from scipy.signal import convolve2d,correlate2d
from cnn import CNN




X = np.array([ [[1,6,2],[5,3,1],[7,0,4]] , [[1,2,3],[4,5,6],[7,8,9]] , [[9,8,7],[6,5,4],[3,2,1]] ])
#k = np.array([[[1,0],[0,0]],[[0,1],[0,0]],[[0,0],[-1,0]]])

C = ConvolutionLayer(3,2,X.shape)
Z = C.forward(X)


print(X)
print(C.kernels)
print(Z)
