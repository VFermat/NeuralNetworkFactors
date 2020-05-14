import math
import os
import sys
from json import dumps
from time import time

import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from helpers.create import createInputs
from helpers.save import saveResults


def buildModel(denses, inputShape, denseSize=20):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=inputShape))
    model.add(LSTM(units=50))
    for i in range(denses - 1):
        model.add(Dense(denseSize))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def run(epochs, denses, xTrain, yTrain, xTest, yTest):
    model = buildModel(denses, (xTrain.shape[1], 1))

    xTrain, xVal, yTrain, yVal = train_test_split(
        xTrain, yTrain, test_size=0.33)
    init = time()
    history = model.fit(xTrain, yTrain, epochs=epochs,
                        validation_data=(xVal, yVal), batch_size=1)
    end = time()

    results = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'run_time': end - init
    }

    yPred = model.predict(xTest)
    results['mse'] = mean_squared_error(yTest, yPred)

    saveResults("LSTM-NOROLLING-FACTORS", epochs, denses, results)

    plt.plot(yTest, label='Real')
    plt.plot(yPred, label='Pred')
    plt.legend()
    plt.show()


def main():
    trainX, trainY, testX, testY = createInputs()
    run(1, 2, trainX, trainY, testX, testY)


if __name__ == '__main__':
    main()
