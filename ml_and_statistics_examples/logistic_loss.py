"""plotting the logistic loss function
when y is equal to one to minimise the loss function a need to be as big as possible in the range of 1, keeping in mind
a is the output of a sigmoid function, hence is nearly 1 for large values of x and is almost zero for large negative
values of x
"""
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_loss(a, y):
    return -(y * np.log(a)) + ((1 - y) * np.log(1 - a))


if __name__ == '__main__':
    X_MAX: float = 1
    vlog_loss = np.vectorize(log_loss)
    x = np.linspace(0.001, X_MAX, 200)

    fig, ax = plt.subplots(figsize=(12, 6))
    y_1 = log_loss(x, 1)
    y_0 = log_loss(x, 0)
    ax.plot(x, y_1, label='y=1')
    ax.plot(x, y_0, label='y=0')
    ax.hlines(y=0, xmin=0, xmax=X_MAX, colors='r', ls='--')
    plt.legend()

    fig, ax = plt.subplots(figsize=(12, 6))
    rng = 10
    x = np.linspace(-rng, rng, 200)
    y_sig = sigmoid(x)
    ax.plot(x, y_sig)
    ax.hlines(y=0.5, xmin=-rng, xmax=rng, colors='r', ls='--')

    plt.show()
