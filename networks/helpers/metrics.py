import keras.backend


def rootMeanSquaredError(yTest, yPred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(yTest - yPred)))
