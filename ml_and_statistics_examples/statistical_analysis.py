from typing import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import collections
from datetime import datetime, timedelta
from pathlib import Path
from importlib import reload
from varname import nameof
import itertools
import functools
import subprocess
import io
import os
import gc
import re
import sys
import time
import logging
import pickle
import json
import random
import string
import requests
from slibtk import slibtk
import configparser
import copy
import shutil

from dstk import dptk, mltk, dviztk
import pyperclip
import numpy as np
import scipy
from scipy import stats
from tqdm.auto import tqdm
import pandas as pd
from sklearn import (
    model_selection,
    metrics,
    preprocessing,
    ensemble,
    neighbors,
    cluster,
    decomposition,
    inspection,
    linear_model
)
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn_extra.cluster import KMedoids
import xgboost as xgb
import lightgbm as lgbm
import optuna
from optuna import Trial
import optuna.integration.lightgbm as lgb
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from functools import partial
import torch
import torch.nn as nn
from pygcp import pygcp
import shap

from google.cloud import storage, bigquery, secretmanager
import seaborn as sns
import plotly.express as px
import warnings

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.5f}'.format)

warnings.simplefilter(action='ignore')


def dict2array(items) -> Tuple[np.array, np.array]:
    array = np.array(list(items))
    return array.reshape(len(items), 1)


titanic = sns.load_dataset('titanic')
iris = sns.load_dataset('iris')

# CENTRAL TENDENCY #####################################################################################################

sample = np.random.randn(200)
np.mean(sample), np.median(sample)

# DISPERSION ###########################################################################################################
# how a variable is spread out around its central tendency ie mean, median, mode etc

# range: max minus the min
rng = max(sample) - min(sample)

# variance
xhat = np.mean(sample)
sse = sum([(x - xhat) ** 2 for x in sample])
var = sse / len(sample)
variance = np.var(sample)
print(variance, var)

# standard deviation
# the square root of the variance and is comparable to the variable units
# - 1 sd = 68%
# - 2 sd = 95%
# - 3 sd = 97.8%


std = np.std(sample)
print(std, np.sqrt(var))
stdlow1 = xhat - std
stdlow2 = stdlow1 - std
stdlow3 = stdlow2 - std
stdhigh1 = xhat + std
stdhigh2 = stdhigh1 + std
stdhigh3 = stdhigh2 + std

std_1 = (sample > stdlow1) & (sample < stdhigh1)
std_2 = (sample > stdlow2) & (sample < stdhigh2)
std_3 = (sample > stdlow3) & (sample < stdhigh3)

# 68% of the values within 1 standard deviation
sum(std_1) / len(sample)
sum(std_2) / len(sample)
sum(std_3) / len(sample)
# half of the values are on one side of the mean
sum(sample < xhat) / len(sample)

df = pd.DataFrame(sample, columns=['points'])
df['cats'] = np.where(std_1, 'std1', 'other')
df.loc[~std_1, 'cats'] = 'std2'
df.loc[~std_2, 'cats'] = 'std3'
fig = px.histogram(df, x='points', color='cats', nbins=70, opacity=0.5)
fig.plot()

sns.stripplot(x='points', y='cats', data=df)
dviztk.plt_fig_save_open()

# quartiles and IQR
tips = sns.load_dataset('tips')
px.histogram(tips, 'tip', nbins=30, **dviztk.hist_kwargs).plot()

# outliers are considered to be 1.5x the interquatiles range from the upper/lower quartile
q_lower, q_upper = np.percentile(tips['tip'], (25, 75))
iqr = q_upper - q_lower
outliers_mask = tips['tip'] > (q_upper + (iqr * 1.5))
tips['outlier'] = outliers_mask.astype(int)
px.histogram(tips, 'tip', color='outlier', opacity=0.7).plot()

# covariance
jt = tips[['tip', 'total_bill']]
px.scatter(jt, 'tip', 'total_bill', **dviztk.scatter_kwargs).plot()

# correlation
# the extent to which to variables are linearly related
covar = np.cov(jt.iloc[:, 0], jt.iloc[:, 1])[0][1]
covar / np.prod(jt.std()), jt.corr().values[0][1]


# SAMPLING METRICS #####################################################################################################


# confidence intervals

def bootstrap_sample(sample, n_boots):
    """"""
    data = []
    for i in range(n_boots):
        bs = {}
        bs['idx'] = i
        bs['points'] = np.random.choice(sample, len(sample))
        bs['mean'] = np.mean(bs['points'])
        data.append(bs)
    return data


x = iris['sepal_length']

x = np.random.choice(x, 100, replace=False)
x = pd.Series(x)

# calculating the standard deviation, standard error and confidence intervals

m = np.mean(x)

std = np.sqrt(sum((x - np.mean(x)) ** 2) / len(x))
np.std(x)

se = std / np.sqrt(len(x))

z = stats.t.ppf(1 - 0.025, len(x))
ci_low, ci_high = m - (z * se), m + (z * se)  # sample mean plus/minus the margin of error ie (z value time standard error)

# boot strapping
bs = bootstrap_sample(x, 3000)

# bootstrapped means
u = [samp['mean'] for samp in bs]
rng = range(len(u))

# the smaller the sample the bigger the confidence interval because
# the standard deviation of the normal distribution increases, see t table
fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
ax.plot(m, 'bo')
ax.plot(np.arange(len(u)), np.ones(len(u)) * ci_low, 'r')
ax.plot(np.arange(len(u)), np.ones(len(u)) * ci_high, 'r')
ax.plot(rng, u, 'o', alpha=0.1)
plt.ylim([4.3, 7.9])
