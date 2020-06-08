import math
import numpy as np
import os
import sys
from json import dumps
from time import time

import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from helpers.create import createInputs
from helpers.metrics import rootMeanSquaredError
from helpers.save import saveResults


def buildModel(denses, inputShape):
    model = Sequential()
    model.add(Dense(inputShape))
    model.add(Dense(2*inputShape))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

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

    results['KerasEvaluate'] = model.evaluate(xTest, yTest)
    yPred = model.predict(xTest)

    return results, np.argmax(yPred, axis=1), model


def main(epochs):
    xTrain, yTrain, xTest, yTest, scaler = createInputs(factors=False, type="dense", tripleBarrier=True)
    yTrainBinary = to_categorical(yTrain)
    yTestBinary = to_categorical(yTest)
    results, yPred, model = run(epochs, 1, xTrain, yTrainBinary, xTest, yTestBinary)
    saveResults("DENSE-NOROLLING-NOFACTORS-TRIPLEBARRIER", epochs, 3, results)


if __name__ == '__main__':
    main(5)
