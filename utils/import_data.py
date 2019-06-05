import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

FILE_PATH = './clean_data.csv';
TRAIN_TO_TEST_RATIO = 0.8
TRAIN_FILE_NAME = './data/train.csv'
TEST_FILE_NAME = './data/test.csv'
CORR_FILE_NAME = './data/corr.png'

data = pn.read_csv(FILE_PATH)
selection = np.random.randn(len(data)) < TRAIN_TO_TEST_RATIO

train = normalize(data[selection])
test = normalize(data[~selection])

np.savetxt(TRAIN_FILE_NAME, train, delimiter=',')
np.savetxt(TEST_FILE_NAME, test, delimiter=',');

# train.to_csv(TRAIN_FILE_NAME)
# test.to_csv(TEST_FILE_NAME)
# plt.matshow(data.corr(), cmap='Blues', interpolation='nearest')
# plt.savefig(CORR_FILE_NAME)
