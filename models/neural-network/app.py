from utils import load_data, path, separate
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import scale, normalize
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
training = load_data('../../data/training.csv')
validation = load_data('../../data/validation.csv')
trainX, trainY = separate(training)
trainX, trainY = normalize(scale(trainX)), normalize(scale(trainY))
testX, testY = separate(validation)
testX, testY = normalize(scale(testX)), normalize(scale(testY))

model = tf.keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(2)
])

opt = tf.keras.optimizers.SGD(lr=0.01)

opt = tf.keras.optimizers.SGD(lr=0.01)

model.compile(
    optimizer = opt,
    loss='mean_squared_error',
    metrics=['accuracy']
)

history = model.fit(trainX, trainY, epochs=80, batch_size=5, validation_data=(testX, testY), verbose=1)

[test_loss, test_acc] = model.evaluate(testX, testY)

# Mean Squared Error
def MSE(y_predicted, y):
    squared_error = (y_predicted - y) ** 2
    sum_squared_error = np.sum(squared_error)
    mse = sum_squared_error / y.size
    return (mse)

