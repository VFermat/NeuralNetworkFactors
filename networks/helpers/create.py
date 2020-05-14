import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def scaleDataFrame(dataFrame):
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataFrame.values)


def createInputs(factors=True, type='lstm', trainSize=0.7):
    dataFrame = pd.read_excel(
        '/Users/eller/insper/iniciacao/Repos/NeuralNetworkFactors/data/dailyStocks.xlsx', index_col=0)
    if not factors:
        dataFrame.drop(['Mkt-RF', 'SMB', 'HML', 'RF'], axis=1)

    data = scaleDataFrame(dataFrame)
    trainSize = int(data.shape[0] * trainSize)

    trainX = data[:trainSize, :-1]
    trainY = data[1:trainSize+1, -1]

    testX = data[trainSize:-1, :-1]
    testY = data[trainSize+1:, -1]

    if type == 'lstm':
        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
        testX = testX.reshape(
            (data.shape[0] - trainX.shape[0]-1, trainX.shape[1], 1))

    return trainX, trainY, testX, testY
