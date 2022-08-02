import numpy as np

class Layer:
    def slide(self,X,window_shape,s=1):
        h1,w1 = X.shape
        h2,w2 = window_shape
        if (h1-h2)%s != 0 or (w1-w2)%s != 0:
            raise Exception('Invalid dimensions ' + str(X.shape) + ',' + str(window_shape))
        for i in range(0,h1-h2+1,s):
            for j in range(0,w1-h2+1,s):
                yield X[i:i+h2,j:j+w2],i,i//s,j,j//s # window,i,zi,j,zj
    def relu(self,X):
        return np.maximum(X,0)
    def softmax(self,Z):
        Z -= np.max(Z) # keep numbers small for numeric stability
        Z = np.exp(Z)
        return Z / np.sum(Z)

class ConvolutionLayer(Layer):
    def __init__(self,n_kernels,n_channels,size,learning_rate=0.01,reg_factor=0.01,momentum_factor=0.9,clip_threshold=1,init_factor=1):
        #self.kernels = self.weights = np.random.randn(n_kernels,n_channels,size,size) * 0.001
        self.kernels = None
        self.n_kernels = n_kernels
        self.n_channels = n_channels
        self.size = size
        self.activation = self.relu
        self.learning_rate = learning_rate
        self.reg_factor = reg_factor
        self.momentum_factor = momentum_factor
        self.clip_threshold = clip_threshold
        self.init_factor = init_factor
        self.vK,self.vB = 0,0
    def convolve(self,X1,X2,pad=0,stride=1):
        X1 = np.pad(X1,((pad,pad),(pad,pad)))
        h1,w1 = X1.shape
        h2,w2 = X2.shape
        Z = np.zeros(((h1-h2)//stride+1,(w1-w2)//stride+1))
        for window,i,zi,j,zj in self.slide(X1,X2.shape,s=stride):
            Z[zi,zj] = np.sum(window * X2)
        return Z
    def forward(self,X):
        self.X = X

        if self.kernels is None:
            self.kernels = self.weights = np.random.randn(self.n_kernels,self.n_channels,self.size,self.size) * np.sqrt(2/X.size) * self.init_factor
            self.biases = np.zeros(self.n_kernels)

        Z = [None]*len(self.kernels)
        for m in range(len(self.kernels)):
            Z[m] = sum(self.convolve(X[c],self.kernels[m][c]) for c in range(len(self.X))) + self.biases[m]
        return self.activation(np.stack(Z))
    def backward(self,dLdZ):
        dLdK = np.zeros(self.kernels.shape)
        dLdB = np.sum(dLdZ,axis=(1,2))
        dLdX = np.zeros(self.X.shape)
        for m in range(len(self.kernels)):
            for c in range(len(self.X)):
                dLdK[m] += self.convolve(self.X[c],dLdZ[m])
                dLdX[c] += self.convolve(dLdZ[m],np.flip(self.kernels[m][c]),pad=len(self.kernels[m][c])-1)

       # if np.max(abs(dLdK)) >= self.clip_threshold:
            #dLdK = (self.clip_threshold / np.max(abs(dLdK))) * dLdK

        self.dz = dLdZ #

        self.vK = self.momentum_factor * self.vK + (1-self.momentum_factor) * dLdK
        self.vB = self.momentum_factor * self.vB + (1-self.momentum_factor) * dLdB
        #self.kernels -= self.learning_rate * dLdK #- self.reg_factor * self.kernels
        #self.biases -= self.learning_rate * dLdB
        self.kernels -= self.learning_rate * self.vK - self.reg_factor * self.kernels
        self.biases -= self.learning_rate * self.vB
        
        return dLdX * (self.X != 0)

class PoolingLayer(Layer):
    def __init__(self,size,stride=None):
        self.size = size
        self.stride = stride or size
        self.weights = 0
    def maxpool(self,X):
        d1,h1,w1 = X.shape
        h2,w2 = self.size,self.size
        Z = np.zeros((d1,(h1-h2)//self.stride+1,(w1-w2)//self.stride+1))
        for k in range(d1):
            for window,i,zi,j,zj in self.slide(X[k],(h2,w2),s=self.stride):
                Z[k,zi,zj] = np.max(window)
        return Z
    def avgpool(self,X):
        pass
    def forward(self,X):
        self.X = X
        return self.maxpool(X)
    def backward(self,dLdZ):
        self.dz = dLdZ #
        dLdX = np.zeros(self.X.shape)
        for k in range(self.X.shape[0]):
            for window,i,zi,j,zj in self.slide(self.X[k],(self.size,self.size),s=self.stride):
                mi,mj = np.unravel_index(np.argmax(window),(self.size,self.size))
                dLdX[k,i+mi,j+mj] = dLdZ[k,zi,zj]

        return dLdX * (self.X != 0)

class DenseLayer(Layer):
    def __init__(self,m_units,learning_rate=0.01,reg_factor=0.01,momentum_factor=0.9,clip_threshold=1,init_factor=1):
        self.m_units = m_units
        self.weights = None
        #self.biases = np.zeros(m_units)
        self.activation = self.relu
        self.learning_rate = learning_rate
        self.reg_factor = reg_factor
        self.momentum_factor = momentum_factor
        self.clip_threshold = clip_threshold
        self.init_factor = init_factor
        self.vW,self.vB = 0,0
    def forward(self,X):
        self.X = X
        if X.ndim > 1:
            X = X.flatten()

        if self.weights is None:
            self.weights = np.random.randn(self.m_units,len(X)) * np.sqrt(2/X.size) * self.init_factor
            self.biases = np.zeros(self.m_units)

        return self.activation(np.dot(self.weights,X) + self.biases)
    def backward(self,dLdZ):
        dLdB = dLdZ
        dLdW = np.outer(dLdZ,self.X.flatten())

        self.dz = dLdZ #

        dLdX = np.dot(self.weights.T,dLdZ).reshape(self.X.shape)

        #if np.max(abs(dLdW)) >= self.clip_threshold:
            #dLdW = (self.clip_threshold / np.max(abs(dLdW))) * dLdW


        #self.velocity = self.velocity * self.momentum_factor - self.learning_rate * dLdW
        #self.weights -= self.learning_rate * dLdW #- self.reg_factor * self.weights
        #self.biases -= self.learning_rate * dLdB
        #self.weights += self.velocity - self.reg_factor * self.weights
        #self.biases += -self.learning_rate * dLdB

        self.vW = self.momentum_factor * self.vW + (1-self.momentum_factor) * dLdW
        self.vB = self.momentum_factor * self.vB + (1-self.momentum_factor) * dLdB
        self.weights -= self.learning_rate * self.vW - self.reg_factor * self.weights
        self.biases -= self.learning_rate * self.vB

        return dLdX * (self.X != 0)

