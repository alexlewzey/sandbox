from typing import List, Tuple, Dict, Callable, Union, Optional, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

x = np.arange(0, 100)
y = 0.5 * x ** 2 + 3 * x + np.random.randint(-50, 50, size=x.shape) * np.random.randint(-50, 50, size=x.shape)


def ewa(a1, a2, b1=0.9):
    return b1 * a2 + (1 - b1) * a1

e = []
v1 = 0
for v2 in y:
    v1 = v1*0.9 + v2*0.1
    e.append(v1)

e3 = []
v1 = 0
for v2 in y:
    v1 = v1*0.5 + v2*0.5
    e3.append(v1)


fig, ax = plt.subplots()
ax.scatter(x, y)
ax.scatter(x, e)
ax.scatter(x, e3)
plt.show()
