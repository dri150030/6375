import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from layers import Layer,ConvolutionLayer,PoolingLayer,DenseLayer
from cnn import CNN


df = pd.read_csv('writing_data.csv')
xs = df.to_numpy()
xs,ys = xs[:,1:].reshape((len(xs),28,28)),xs[:,0]
xs = np.reshape(StandardScaler().fit_transform(np.reshape(xs,(-1,len(xs)))),xs.shape)

chars = [chr(ord('A')+i) for i in range(26)]

net = CNN((('c',16,1,5),('p',2),('d',26)),
        labels=chars,max_iter=3000,learning_rate=0.003,reg_factor=0.00001,momentum_factor=0.9,clip_threshold=3,init_factor=0.001)

accuracy = net.fit(xs,ys)

print(net.predict(xs[23489]))
print(chars[ys[23489]])

with open('model','wb') as f:
    import pickle
    pickle.dump(net,f)
