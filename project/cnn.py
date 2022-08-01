import numpy as np
from functools import reduce
from random import choice
from layers import ConvolutionLayer,PoolingLayer,DenseLayer

class CNN:
    def __init__(self,hidden_layers,reg_factor=0.1,learning_rate=0.1,max_iter=100,labels=None):
        self.reg_factor = reg_factor
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.labels = labels

        # net = CNN((('c',3,3,3),('p',2),('d',10))) # 3x3x3x3 conv layer, 2x2 pooling layer, 10-neuron dense layer
        self.net = []
        for layer in hidden_layers:
            if layer[0] == 'c':
                self.net.append(ConvolutionLayer(*layer[1:],learning_rate=self.learning_rate))
            elif layer[0] == 'p':
                self.net.append(PoolingLayer(*layer[1:]))
            elif layer[0] == 'd':
                self.net.append(DenseLayer(*layer[1:],learning_rate=self.learning_rate))
        self.net[-1].activation = self.net[-1].softmax # final layer uses softmax activation

    def reg_loss(self):
        r = 0
        for layer in self.net:
            r += self.reg_factor * np.sum(layer.weights**2)
        return r
    def loss(self,P,y):
        return -np.log(P[y]) + self.reg_loss() # cross entropy loss = -ln(predicted_probability(true class))
    def forward(self,X):
        if X.ndim == 2:
            X = X[None,:]
        elif X.ndim != 3:
            raise Exception('Invalid input dimensions')
        self.P = reduce(lambda x,unit: unit.forward(x),self.net,X)
        self.y_pred = np.argmax(self.P)
    def backward(self,y):
        dLdZ = np.array([p if i != y else p-1 for i,p in enumerate(self.P)])
        dLdX = reduce(lambda dL,unit: unit.backward(dL),reversed(self.net),dLdZ)
    def fit(self,xs,ys):
        from collections import deque
        self.accuracy,self.correct = [],deque(maxlen=100)
        for e in range(self.max_iter):
            i = choice(range(len(xs)))
            X,y = xs[i],ys[i]
            self.forward(X)
            self.backward(y)
            self.correct.append(1 if y == self.y_pred else 0)
            self.accuracy.append(sum(self.correct) / 100)
            if False:
                print('------------------------------------')
                print('P: ' + str(self.P))
                print('ypred,ytrue: ' + str((self.y_pred,y)))
                print('iter,acc,loss: ' + str((e,self.accuracy[-1],self.loss(self.P,y))))
                input()
            elif e%10 == 0:
                print((e,self.accuracy[-1]))
        return self.accuracy
    def predict(self,X):
        self.forward(X)
        return self.labels[self.y_pred]
    def probability(self,x,y):
        return self.P
