import numpy as np
from layers import ConvolutionLayer,PoolingLayer,DenseLayer
#from sklearn.metrics import log_loss

class CNN:
    def __init__(self,convolution_layers=None,dense_layers=(100,),l2_factor=0.1,learning_rate=0.1,batch_size=1):
        pass
    def l2reg(self,w):
        return self.l2f * np.sum(w**2)
    def loss(self,P,y):
        return -np.log(P[y]) #+ l2reg() # cross entropy loss = -ln(predicted_probability(true class))
    def softmax(self,Z):
        Z -= np.max(Z) # keep numbers small for numeric stability
        Z = np.exp(Z)
        return Z / np.sum(Z)
    def forward(self,X):
        self.Z = reduce(self.net)
        self.P = self.softmax(self.Z)
        self.y_pred = np.argmax(self.P)
    def backward(self,P,y):
        dLdZ = np.array([p if i != y else p-1 for i,p in enumerate(P)])
        return dLdZ

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
