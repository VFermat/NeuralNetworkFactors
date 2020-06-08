import math
import os
import sys
from json import dumps
from time import time

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from helpers.create import createInputs
from helpers.metrics import rootMeanSquaredError
from helpers.save import saveResults

TRAINING_WINDOW = 10


def buildModel(denses, inputShape, denseSize=20):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=inputShape))
    model.add(LSTM(units=50))
    for i in range(denses - 1):
        model.add(Dense(denseSize))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

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

    results['KerasEvaluate'] = model.evaluate(xTest, yTest)
    yPred = model.predict(xTest)

    return results, np.argmax(yPred, axis=1), model


def main(epochs):
    xTrain, yTrain, xTest, yTest, scaler = createInputs(factors=False, tripleBarrier=True)
    yTrainBinary = to_categorical(yTrain)
    yTestBinary = to_categorical(yTest)
    results, yPred, model = run(epochs, 1, xTrain, yTrainBinary, xTest, yTestBinary)
    saveResults("LSTM-NOFACTORS-TRIPLEBARRIER", epochs, 2, results)


if __name__ == '__main__':
    main(1)
