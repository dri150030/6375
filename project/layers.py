import numpy as np
#from scipy.signal import convolve2d,correlate2d
#from sklearn.metrics import log_loss

relu = lambda x: np.maximum(x,0,out=x)

class Layer:
    def __init__(self):
        pass
    def slide(self,X):
        pass
    def relu(self):
        pass


class ConvolutionLayer:
    def __init__(self,n_kernels,size,xshape):
        self.kernels = np.random.randn(n_kernels,xshape[0],size,size)
        self.biases = np.random.randn(n_kernels)
        #self.Z = np.zeros((n_kernels,(xshape[1]-size)+1,(xshape[2]-size)+1))
    def convolve(self,X1,X2,pad=0,stride=1):
        X1 = np.pad(X1,((0,0),(pad,pad),(pad,pad)))
        d1,h1,w1 = X1.shape
        d2,h2,w2 = X2.shape
        if d1 != d2 or (h1-h2)%stride != 0 or (w1-w2)%stride != 0:
            raise Exception('Invalid stride')
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
    def backward(self,dLdZ):#,learning_rate):
        dLdK = self.convolve(np.repeat(self.X,len(dLdZ),axis=0),dLdZ)
        dLdB = np.sum(dLdZ,axis=(1,2))
        # pad for full convolution
        dLdX = sum(self.convolve(np.flip(K,axis=(1,2)),dLdZi[None,:],pad=len(dLdZ[0])-1)
                for K,dLdZi in zip(self.kernels,dLdZ))
        #self.kernels -= learning_rate * dLdK
        #self.biases -= learning_rate * dLdB
        return dLdX


class PoolingLayer:
    def __init__(self,size,stride=None):
        self.size = size
        self.stride = stride or size
    def maxpool(self,X):
        d1,h1,w1 = X.shape
        h2,w2 = self.size,self.size
        if (h1-h2)%self.stride != 0 or (w1-w2)%self.stride != 0:
            raise Exception('Invalid stride')
        Z = np.zeros((d1,(h1-h2)//self.stride+1,(w1-w2)//self.stride+1))
        for zi in range(len(Z[0])):
            for zj in range(len(Z[0][zi])):
                i,j = zi*self.stride,zj*self.stride
                Z[:,zi,zj] = np.max(X[:,i:i+h2,j:j+w2],axis=(1,2))
        return Z
    def avgpool(self,X):
        pass
    def forward(self,X):
        self.X = X
        return self.maxpool(X)
    def backward(self,dLdZ):
        dLdX = np.zeros(self.X.shape)
        for k in range(len(dLdZ)):
            for zi in range(len(dLdZ[k])):
                for zj in range(len(dLdZ[k][zi])):
                    i,j = zi*self.stride,zj*self.stride
                    mi,mj = np.unravel_index(np.argmax(self.X[k,i:i+self.size,j:j+self.size]),(self.size,self.size))
                    dLdX[k,i+mi,j+mj] = dLdZ[k,zi,zj]
        return dLdX

class DenseLayer:
    def __init__(self,m,n):
        self.weights = np.random.uniform(-1,1,(m,n))
        self.biases = 1
    def forward(self,X):
        self.xshape = X.shape
        if X.ndim > 1:
            X = X.flatten()
        self.X = X
        return relu(np.dot(self.weights,X))
    def backward(self,dLdZ):
        dLdB = dLdZ
        dLdW = np.outer(dLdZ,self.X)
        dLdX = np.dot(self.weights.T,dLdZ).reshape(self.xshape)
        return dLdX

