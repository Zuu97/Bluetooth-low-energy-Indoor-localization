import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow import keras
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Input, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML
import numpy as np
from PIL import Image

import logging
logging.getLogger('tensorflow').disabled = True

from util import cnn_data, eucledian_distance, custom_loss
from variables import*
# np.random.seed(seed)

class ITScnn(object):
    def __init__(self):
        Xtrain, Ytrain, Xtest, Ytest = cnn_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest  = Xtest
        self.Ytest  = Ytest
        self.callback = EarlyStopping(
                                monitor='val_loss',
                                patience=20, verbose=0,
                                mode='auto',
                                restore_best_weights=True
                                )

    def classifier(self):
        inputs = Input(shape=(self.Xtrain.shape[1], self.Xtrain.shape[2], 1))
        x = Conv2D(3, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(inputs)
        x = MaxPooling2D(2)(x)
        x = Conv2D(6, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)
        x = MaxPooling2D(2)(x)
        x = Conv2D(12, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)
        outputs = Dense(2, activation='relu')(Flatten()(x))
        self.model = Model(
                        inputs=inputs,
                        outputs=outputs
                        )
        self.model.summary()

    def train(self):
        self.classifier()
        self.model.compile(
                    loss=custom_loss,
                    optimizer=Adam(learning_rate),
                    metrics=['accuracy']
                    )
        self.history = self.model.fit(
                            self.Xtrain,
                            self.Ytrain,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_data=[self.Xtest, self.Ytest],
                            # callbacks = [self.callback]
                            )

    def save_model(self):
        self.model.save(cnn_weights)

    def load_model(self):
        loaded_model = load_model(cnn_weights)
        loaded_model.compile(
                    loss=custom_loss,
                    optimizer=Adam(learning_rate),
                    metrics=['accuracy']
                    )
        self.model = loaded_model

    def eculedian_error(self):
        Ypred = self.model.predict(self.Xtest)
        Xcoor = self.Ytest[:, 0]
        Ycoor = self.Ytest[:, 1]

        PredXcoor = Ypred[:, 0]
        PredYcoor = Ypred[:, 1]

        l2dists_mean, l2dists = eucledian_distance((PredXcoor, PredYcoor), (Xcoor, Ycoor))
        print("Mean distance error : {}".format(l2dists_mean))

        sortedl2_deep = np.sort(l2dists)
        prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)
        fig, ax = plt.subplots()
        lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')
        plt.title('CDF of Euclidean distance error for CNN')
        plt.xlabel('Distance (m)')
        plt.ylabel('Probability')
        plt.grid(True)
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle('-.')

        plt.savefig(cdf_error_cnn, dpi=300)
        plt.show()
        plt.close()

    def run(self):
        if os.path.exists(cnn_weights):
            print("Model Loading !!")
            self.load_model()
        else:
            print("Model Training !!")
            self.classifier()
            self.train()
            # self.save_model()

        self.eculedian_error()

if __name__ == "__main__":
    model = ITScnn()
    model.run()