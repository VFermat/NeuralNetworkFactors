import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def scaleDataFrame(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def createInputs(factors=True, tripleBarrier=False, type='lstm', trainSize=0.7):
    dataFrame = pd.read_excel(
        '/Users/eller/insper/iniciacao/Repos/NeuralNetworkFactors/data/dailyStocks.xlsx', index_col=0)
    if not factors:
        dataFrame.drop(['Mkt-RF', 'SMB', 'HML', 'RF'], axis=1)

    data = dataFrame.values
    x, _ = scaleDataFrame(data[:, :-1])
    trainSize = int(data.shape[0] * trainSize)

    if tripleBarrier:
        y = buildBarriers(dataFrame, 'SP500Price').values
        yTest = y[trainSize+1:]
        yTrain = y[1:trainSize+1]
        scaler = ''
    else:
        y, scaler = scaleDataFrame(data[:, -1].reshape((-1, 1)))
        yTest = y[trainSize+1:, 0]
        yTrain = y[1:trainSize+1, 0]

    xTrain = x[:trainSize, :]
    xTest = x[trainSize:-1, :]

    if type == 'lstm':
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
        xTest = xTest.reshape(
            (data.shape[0] - xTrain.shape[0]-1, xTrain.shape[1], 1))

    return xTrain, yTrain, xTest, yTest, scaler



def calculateSignal(row, windowSize):
    if row.isna().values.any():
        return 0
    actualPrice = row.P
    pastKeys = [f'P-{i}' for i in range(1, windowSize + 1)] + ['P']
    futureKeys = [f'P+{i}' for i in range(1, windowSize + 1)]
    superiorBarrier = actualPrice + row[pastKeys].std()
    inferiorBarrier = actualPrice - row[pastKeys].std()
    try:
        supBreakpoint = list(row[futureKeys] > superiorBarrier).index(True)
    except:
        supBreakpoint = -1
    try:
        infBreakpoint = list(row[futureKeys] < inferiorBarrier).index(True)
    except:
        infBreakpoint = -1
    if infBreakpoint == -1 and supBreakpoint == -1:
        return 0
    elif infBreakpoint < supBreakpoint:
        return 1
    else:
        return 2

def buildBarriers(df, stock, windowSize=7):
    temp = df[[stock]]
    for i in range(1, windowSize + 1):
        temp[f'P+{i}'] = temp[stock].shift(-i)
        temp[f'P-{i}'] = temp[stock].shift(i)

    temp['P'] = temp[stock]
    return temp.apply(calculateSignal, axis=1, args=(windowSize, ))
