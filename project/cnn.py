import numpy as np
from functools import reduce
from layers import ConvolutionLayer,PoolingLayer,DenseLayer

class CNN:
    def __init__(self,hidden_layers,reg_factor=0.1,learning_rate=0.1):
        # net = CNN((('c',3,3,3),('p',2),('d',10))) # 3x3x3x3 conv layer, 2x2 pooling layer, 10-neuron dense layer
        self.net = []
        for layer in hidden_layers:
            if layer[0] == 'c':
                self.net.append(ConvolutionLayer(*layer[1:]))
            elif layer[0] == 'p':
                self.net.append(PoolingLayer(*layer[1:]))
            elif layer[0] == 'd':
                self.net.append(DenseLayer(*layer[1:]))
        self.learning_rate = learning_rate
        self.reg_factor = reg_factor
    def l2reg(self,w):
        return self.l2f * np.sum(w**2)
    def loss(self,P,y):
        return -np.log(P[y]) #+ l2reg() # cross entropy loss = -ln(predicted_probability(true class))
    def softmax(self,Z):
        Z -= np.max(Z) # keep numbers small for numeric stability
        Z = np.exp(Z)
        return Z / np.sum(Z)
    def forward(self,X):
        self.Z = reduce(lambda x,unit: unit.forward(x),self.net,X)
        self.P = self.softmax(self.Z)
        self.y_pred = np.argmax(self.P)
    def backward(self):
        dLdZ = np.array([p if i != self.y_pred else p-1 for i,p in enumerate(self.P)])
        dLdX = reduce(lambda dL,unit: unit.backward(dL),reversed(self.net),dLdZ)
    def fit(self,xs,ys):
        if x.ndim == 2:
            pass
        elif x.ndim != 3:
            raise Exception
    def predict(self,X):
        self.forward(X)
        return self.labels[self.y_pred]
    def probability(self,x,y):
        return self.Z
