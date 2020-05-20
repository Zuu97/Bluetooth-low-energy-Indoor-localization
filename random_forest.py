from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
from variables import*
from util import rf_data
import joblib

class ITSrandomForest(object):
    def __init__(self):
        Xtrain, Ytrain, Xtest, Ytest = rf_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest  = Xtest
        self.Ytest  = Ytest

    def classifier(self, depth):
        model = RandomForestClassifier(
                                max_depth=depth,
                                random_state=seed)
        model.fit(self.Xtrain, self.Ytrain)
        self.model = model

    def prediction(self):
        Ptrain = self.model.predict(self.Xtrain)
        Ptest  = self.model.predict(self.Xtest)

        train_acc = round(np.mean(Ptrain==self.Ytrain), 3)
        test_acc  = round(np.mean(Ptest==self.Ytest), 3)
        return train_acc, test_acc

    def grid_search(self):
        for depth in range(1,100,5):
            self.classifier(depth)
            train_acc, test_acc = self.prediction()
            print("max depth : {}, Train accuracy : {}, Test accuracy : {}".format(depth, train_acc, test_acc))

if __name__ == "__main__":
    model = ITSrandomForest()
    model.grid_search()

