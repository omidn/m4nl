import numpy as np
import pandas as pa

TRAIN_FILE = './data/train.csv'
TEST_FILE = './data/test.csv';
train, test = pa.read_csv(TRAIN_FILE), pa.read_csv(TEST_FILE)

def kernel(first, second, param):
    sqdist = np.sum(first ** 2,1).reshape(-1,1) + np.sum(second ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-.5 * (1 / param) * sqdist)

param = 0.1
K_ss = kernel(Xtest, Xtest, param)
