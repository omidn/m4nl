import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

FILE_PATH = './clean_data.csv';
TRAIN_TO_TEST_RATIO = 0.8
TRAIN_FILE_NAME = './data/train.csv'
TEST_FILE_NAME = './data/test.csv'
CORR_FILE_NAME = './data/corr.png'

def norm(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

data = pn.read_csv(FILE_PATH)
selection = np.random.randn(len(data)) < TRAIN_TO_TEST_RATIO


train = norm(data[selection])
test = norm(data[~selection])

np.savetxt(TRAIN_FILE_NAME, train, delimiter=',')
np.savetxt(TEST_FILE_NAME, test, delimiter=',');

# train.to_csv(TRAIN_FILE_NAME)
# test.to_csv(TEST_FILE_NAME)
# plt.matshow(data.corr(), cmap='Blues', interpolation='nearest')
# plt.savefig(CORR_FILE_NAME)
