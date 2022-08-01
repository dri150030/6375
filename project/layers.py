import numpy as np

class Layer:
    def slide(self,X,window_shape,s=1):
        d,h1,w1 = X.shape
        h2,w2 = window_shape
        if (h1-h2)%s != 0 or (w1-w2)%s != 0:
            raise Exception('Invalid dimensions')
        for i in range(0,h1-h2+1,s):
            for j in range(0,w1-h2+1,s):
                yield X[:,i:i+h2,j:j+w2],i//s,j//s
    def relu(self,X):
        return np.maximum(X,0)

class ConvolutionLayer(Layer):
    def __init__(self,n_kernels,n_channels,size,learning_rate=0.1):
        self.kernels = np.random.randn(n_kernels,n_channels,size,size)
        self.biases = np.random.randn(n_kernels)
        self.learning_rate = learning_rate
    def convolve(self,X1,X2,pad=0,stride=1):
        X1 = np.pad(X1,((0,0),(pad,pad),(pad,pad)))
        d1,h1,w1 = X1.shape
        d2,h2,w2 = X2.shape
        Z = np.zeros(((h1-h2)//stride+1,(w1-w2)//stride+1))
        for window,i,j in self.slide(X1,(h2,w2),s=stride):
            Z[i,j] = np.sum(window * X2)
        return Z
    def forward(self,X):
        self.X = X
        Z = np.stack([self.convolve(X,K) + b for K,b in zip(self.kernels,self.biases)])
        return self.relu(Z)
    def backward(self,dLdZ):
        dLdK = np.zeros(self.kernels.shape)
        dLdB = np.sum(dLdZ,axis=(1,2))
        dLdX = np.zeros(self.X.shape)
        for m in range(len(self.kernels)):
            dLdK[m] = self.convolve(self.X,dLdZ[m][None,:])
            dLdX += self.convolve(np.flip(self.kernels[m],axis=(1,2)),dLdZ[m][None,:],pad=len(dLdZ[m])-1)
        self.kernels -= self.learning_rate * dLdK
        self.biases -= self.learning_rate * dLdB
        return dLdX

class PoolingLayer(Layer):
    def __init__(self,size,stride=None):
        self.size = size
        self.stride = stride or size
    def maxpool(self,X):
        d1,h1,w1 = X.shape
        h2,w2 = self.size,self.size
        Z = np.zeros((d1,(h1-h2)//self.stride+1,(w1-w2)//self.stride+1))
        for window,i,j in self.slide(X,(h2,w2),s=self.stride):
            Z[:,i,j] = np.max(window,axis=(1,2))
        return Z
    def avgpool(self,X):
        pass
    def forward(self,X):
        self.X = X
        return self.maxpool(X)
    def backward(self,dLdZ):
        dLdX = np.zeros(self.X.shape)
        for window,i,j in self.slide(self.X,(self.size,self.size),s=self.stride):
            for k in range(len(window)):
                mi,mj = np.unravel_index(np.argmax(window[k]),(self.size,self.size))
                dLdX[k,i+mi,j+mj] = dLdZ[k,i,j]
        return dLdX

class DenseLayer(Layer):
    def __init__(self,m_units,learning_rate=0.1):
        self.m_units = m_units
        self.weights = None
        self.biases = np.random.randn(m_units)
        self.learning_rate = learning_rate
    def forward(self,X):
        self.xshape = X.shape
        if X.ndim > 1:
            X = X.flatten()
        self.X = X
        if self.weights == None:
            self.weights = np.random.randn(self.m_units,len(X))
        return self.relu(np.dot(self.weights,X) + self.biases)
    def backward(self,dLdZ):
        dLdB = dLdZ
        dLdW = np.outer(dLdZ,self.X)
        dLdX = np.dot(self.weights.T,dLdZ).reshape(self.xshape)
        self.weights -= self.learning_rate * dLdW
        self.biases -= self.learning_rate * dLdB
        return dLdX

