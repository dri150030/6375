import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

class NeuralNet:
    def __init__(self,url,inputs,output,test_ratio=0.2):
        # read csv from url and read feature/output cols into numpy arrays
        df = pd.read_csv(url)
        self.X_labels,self.y_label = inputs,output
        self.X,self.y = df[inputs].to_numpy(),df[[output]].to_numpy()

        # scale and split data
        self.scaler = StandardScaler()
        self.resplit(test_ratio)

    # generate new train/test split
    def resplit(self,test_ratio=0.2):
        self.X_train_raw,self.X_test_raw,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=test_ratio)
        self.X_train = self.scaler.fit_transform(self.X_train_raw)
        self.X_test = self.scaler.transform(self.X_test_raw)

    def tune(self,params,n):
        # take list of param choices and generate list of param keyword arg dictionaries
        params = [[(k,v) for v in vs] for k,vs in params.items()]
        params = list(map(dict,product(*params)))
        train_loss,test_loss,epochs = [0]*len(params),[0]*len(params),[0]*len(params)
        print('Tuning',self.__class__.__name__,'parameters...')
        for t in range(n):
            print('Tuning iteration',t+1,'of',n,'...')
            # generate new data split in each iteration
            self.data.resplit()
            for i,kwargs in enumerate(params):
                model = self.mclass(**kwargs)
                model.fit(self.data.X_train,self.data.y_train)
                y_train_pred = model.predict(self.data.X_train)
                y_test_pred = model.predict(self.data.X_test)

                # log train/test loss and epochs
                train_loss[i] += mean_squared_error(self.data.y_train,y_train_pred) / n
                test_loss[i] += mean_squared_error(self.data.y_test,y_test_pred) / n
                epochs[i] += model.actual_epochs / n
        
        # zip params, error, and epochs into log and return the best param
        self.log = sorted(zip(params,train_loss,test_loss,epochs),key=lambda x: x[2])
        self.model = MLPClassifier(**self.log[0][0])

    # TODO: Train and evaluate models for all combinations of parameters
    # specified. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0

if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/dri150030/6375/main/beans.csv'
    neural_network = NeuralNet(url,[],[])
    #neural_network.train_evaluate()
