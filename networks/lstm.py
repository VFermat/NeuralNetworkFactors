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
from helpers.metrics import rootMeanSquaredError
from helpers.save import saveResults


def buildModel(denses, inputShape, denseSize=20):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=inputShape))
    model.add(LSTM(units=50))
    for i in range(denses - 1):
        model.add(Dense(denseSize))
    model.add(Dense(1))
    model.compile(loss=rootMeanSquaredError, optimizer='adam')

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
    results['scaledMse'] = mean_squared_error(yTest, yPred)

    return results, yPred


def main(epochs):
    xTrain, yTrain, xTest, yTest, scaler = createInputs(factors=False)
    results, yPred = run(epochs, 2, xTrain, yTrain, xTest, yTest)
    results['unscaledMse'] = mean_squared_error(
        scaler.inverse_transform(yTest.reshape((-1, 1))), scaler.inverse_transform(yPred))
    saveResults("LSTM-NOROLLING-NOFACTORS", epochs, 2, results)


if __name__ == '__main__':
    main(1)
