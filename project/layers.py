import numpy as np
#from scipy.signal import convolve2d,correlate2d
#from sklearn.metrics import log_loss

relu = lambda x: np.maximum(x,0,out=x)

class ConvolutionLayer:
    def __init__(self,n_kernels,size,xshape):
        self.kernels = np.random.randn(n_kernels,xshape[0],size,size)
        self.biases = np.random.randn(n_kernels)
        #self.Z = np.zeros((n_kernels,(xshape[1]-size)+1,(xshape[2]-size)+1))
    def convolve(self,X1,X2,pad=0,stride=1):
        X1 = np.pad(X1,((0,0),(pad,pad),(pad,pad)))
        d1,h1,w1 = X1.shape
        d2,h2,w2 = X2.shape
        if (h1-h2)%stride != 0 or (w1-w2)%stride != 0:
            raise Exception('Invalid convolution')
        z = np.zeros(((h1-h2)//stride+1,(w1-w2)//stride+1))
        for zi in range(len(z)):
            for zj in range(len(z[zi])):
                i,j = zi*stride,zj*stride
                z[zi,zj] = np.sum(X1[:,i:i+h2,j:j+w2] * X2)
        return z
    def forward(self,X):
        self.X = X
        Z = np.stack([self.convolve(X,K) + b for K,b in zip(self.kernels,self.biases)])
        return relu(Z)
    def backward(self,dLdZ):
        dLdK = convolve(self.X,dLdZ)
        dLdB = dLdZ
        dLdX = convolve(np.flip(self.K),dLdZ,p=len(dLdZ)-1) # pad for full convolution
        return dLdX


class PoolingLayer:
    def maxpool(self):
        pass
    def forward(self,x):
        pass
    def backward(self,x):
        pass

class DenseLayer:
    def __init__(self,m,n):
        self.weights = np.random.uniform(-1,1,(m,n))
        self.biases = 1
    def softmax(self,x):
        pass
    def forward(self,x):
        return relu(np.dot(self.weights,x))
    def backward(self,x):
        pass

