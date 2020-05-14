import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def scaleDataFrame(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def createInputs(factors=True, type='lstm', trainSize=0.7):
    dataFrame = pd.read_excel(
        '/Users/eller/insper/iniciacao/Repos/NeuralNetworkFactors/data/dailyStocks.xlsx', index_col=0)
    if not factors:
        dataFrame.drop(['Mkt-RF', 'SMB', 'HML', 'RF'], axis=1)

    data = dataFrame.values
    x, _ = scaleDataFrame(data[:, :-1])
    y, scaler = scaleDataFrame(data[:, -1].reshape((-1, 1)))
    trainSize = int(data.shape[0] * trainSize)

    xTrain = x[:trainSize, :]
    yTrain = y[1:trainSize+1, 0]

    xTest = x[trainSize:-1, :]
    yTest = y[trainSize+1:, 0]

    if type == 'lstm':
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
        xTest = xTest.reshape(
            (data.shape[0] - xTrain.shape[0]-1, xTrain.shape[1], 1))

    return xTrain, yTrain, xTest, yTest, scaler
