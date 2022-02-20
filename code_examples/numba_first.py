import time

from tqdm import tqdm
from typing import List, Tuple, Dict, Callable, Union, Optional, Any
import numpy as np
import typing
from numba import njit

import numpy as np
import timeit

# %%
nrows = 100_000

@njit()
def test_np():
    a = np.zeros((nrows, 3))
    for i in range(nrows):
        a[i] = np.random.randint(1, 5, size=(3))


@njit()
def test_ls():
    a = []
    for i in range(nrows):
        a.append(np.random.randint(1, 5, size=(3)))


nruns = 3
print(min(timeit.repeat(test_np, number=nruns)))
print(min(timeit.repeat(test_ls, number=nruns)))


array = np.random.random((300_000))


# %%
def make_ts_features(array, Tx: int, Ty: int = 1):
    """
    split a series into ~ N / T test train splits. example of one of the splits [t0, t1, ... tT-1] [tT]
    Args:
        array: a series to be transformed into features
        Tx: number of time steps in features
        Ty: number of time steps in target

    Returns:
        X: training data with windows of size Tx
        y: target data with windows of size Ty
    """
    if Tx > len(array):
        raise ValueError('Window length cannot be longer than series...')
    nsplits: int = len(array) - Tx - Ty + 1
    X = np.zeros((nsplits, Tx))
    y = np.zeros((nsplits, Ty))
    for i in range(nsplits):
        X[i] = array[i:i + Tx]
        y[i] = array[i + Tx: i + Tx + Ty]
    X = X.reshape(*X.shape, 1)
    y = y.reshape(*y.shape, 1)
    return X, y


print('started function')
start = time.time()
X, y = make_ts_features(array, 100, 30)
print(f'time taken: {time.time() - start}')

print('started function')
start = time.time()
X, y = make_ts_features(array, 100, 30)
print(f'time taken: {time.time() - start}')
