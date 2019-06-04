import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

FILE_PATH = './clean_data.csv';
TRAIN_TO_TEST_RATIO = 0.8
TRAIN_FILE_NAME = './data/train.csv'
TEST_FILE_NAME = './data/test.csv'

data = pn.read_csv(FILE_PATH)

selection = np.random.randn(len(data)) < TRAIN_TO_TEST_RATIO

train = data[selection]
test = data[~selection]

train.to_csv(TRAIN_FILE_NAME)
test.to_csv(TEST_FILE_NAME)

plt.matshow(data.corr(), cmap='Blues', interpolation='nearest')
plt.show()
