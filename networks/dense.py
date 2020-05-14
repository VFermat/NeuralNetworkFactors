import math
import os
import sys
from json import dumps
from time import time

import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from helpers.create import createInputs
from helpers.save import saveResults


def buildModel(denses, inputShape):
    model = Sequential()
    model.add(Dense(inputShape))
    model.add(Dense(2*inputShape))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def run(epochs, denses, xTrain, yTrain, xTest, yTest):
    model = buildModel(denses, xTrain.shape[1])

    xTrain, xVal, yTrain, yVal = train_test_split(
        xTrain, yTrain, test_size=0.33)
    init = time()
    history = model.fit(xTrain, yTrain, epochs=epochs,
                        validation_data=(xVal, yVal), batch_size=1, verbose=2)
    end = time()

    results = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'run_time': end - init
    }

    yPred = model.predict(xTest)
    results['mse'] = mean_squared_error(yTest, yPred)

    saveResults("DENSE-NOROLLING-NOFACTORS", epochs, denses, results)

    plt.plot(yTest, label='Real')
    plt.plot(yPred, label='Pred')
    plt.legend()
    plt.show()


def main(epochs):
    trainX, trainY, testX, testY = createInputs(factors=False, type='dense')
    run(epochs, 3, trainX, trainY, testX, testY)


if __name__ == '__main__':
    main(5)
