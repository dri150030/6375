import numpy as np
from layers import ConvolutionLayer,PoolingLayer,DenseLayer
#from sklearn.metrics import log_loss

class CNN:
    def __init__(self,convolution_layers,dense_layers=(100,),l2_factor=0.1,learning_rate=0.1,batch_size=1):
        pass
    def l2reg(self,w):
        return self.l * np.sum(w**2)
    def loss(self,yp,ya):
        # mean squared error
        # log loss / cross entropy
        pass
    def softmax(self,x):
        pass
    def fit(self,x,y):
        if x.ndim == 2:
            pass
        elif x.ndim != 3:
            raise Exception
    def predict(self,x,y):
        pass
    def probability(self,x,y):
        pass
