from utils import load_data, path, separate
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

training = load_data('../../data/training.csv')
validation = load_data('../../data/validation.csv')
trainX, trainY = separate(training)
testX, testY = separate(validation)
trainX /= np.max(trainX)

eta = 0.5

def predict(x, w, b):
    return np.matmul(x, w.T) + b

def l2loss(x, y_true, w, b):
    y_pred = predict(x, w, b)
    loss = y_true - y_pred
    total_loss = np.sum(np.square(y_true - y_pred))
    
    total_dw = -2 * np.matmul(x.T, loss).T
    total_dw /= x.shape[0]
    total_db = -2 * np.mean(loss, axis=0)

    return (np.power(total_loss, 0.5), total_dw, total_db)

def train(x, y, w, b, eta=0.1):
    losses = []
    iters_count = 0
    while True:
        loss, dw, db = l2loss(x, y, w, b)

        w += dw * -eta
        b += db * -eta

        losses.append(loss)

        iters_count += 1

        if (np.sum(np.abs(dw)) <  tolerance):
            break

        if (iters_count % 10000 == 0):
            print ('loss', loss)

    return (w, b, losses, iters_count)

W0 = np.random.rand(2, trainX.shape[1])
b0 = np.random.rand(2)

tolerance = 1e-3
W, b, l, iters = train(trainX, trainY, W0, b0)
print ('learned')
l = np.array(l)
fig, ax = plt.subplots()
ax.plot(np.arange(800, iters), l[800:])
ax.set(
    xlabel='# of iterations',
    ylabel='log10 of loss',
    title='Loss decay over iterations'
)
plt.grid()
plt.show()
