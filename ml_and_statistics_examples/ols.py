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
import warnings

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.5f}'.format)

warnings.simplefilter(action='ignore')

from scipy import stats
import scipy

# regression modules
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

# model selection/evaluaiton
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# pre-processing modules
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA


def dict2array(items) -> Tuple[np.array, np.array]:
    array = np.array(list(items))
    return array.reshape(len(items), 1)


def get_reg_metrics(y_true, pred, pred_nm) -> List[str]:
    """
    return a list of formatted regression metrics
    """
    r2 = r2_score(y, pred).round(3)
    rmse = np.sqrt(mean_squared_error(y, pred)).round(3)
    mae = mean_absolute_error(y, pred).round(3)
    return [
        f'pred_nm: {pred_nm}',
        f'r2: {r2}',
        f'rmse: {rmse}',
        f'mae: {mae}',
        f'std_y: {round(np.std(y), 3)}',
        f'std_pred: {round(np.std(pred), 3)}', f'rmse/std_y: {round(rmse / np.std(pred), 3)}',
    ]


def show_padded_matrix(data: List[List], padding=2) -> None:
    """
    print a padded matrix of a list of lists where each list is a row
    """
    col_lens = [max((len(word) + padding for word in row)) for row in list(zip(*data))]
    for row in metrics:
        row_padded = [word.ljust(ln) for word, ln in zip(row, col_lens)]
        print(''.join(row_padded))


# creating toy data
data = {x: 5 + x ** 2 for x in range(11)}
data = {k: v / 10 for k, v in data.items()}
avg = [np.mean(list(data.values())) for i in data]
test = [1 + 1.2 * x for x in data.keys()]

df = pd.DataFrame(list(data.items()), columns=['x', 'y'])
X, y = df[['x']], df['y']

# fitting a linear model
lm = LinearRegression()
lm.fit(X, y)
pred = lm.predict(X)

X['x2'] = X ** 2
lm.fit(X, y)
pred_d2 = lm.predict(X)

# at any given point it could be 11% of the values
# at any given point it could be around 30% of the values
preds = {'pred': pred, 'avg': avg, 'test': test, 'pred_d2': pred_d2}
metrics = [get_reg_metrics(y, v_pred, k_nm) for k_nm, v_pred in preds.items()]
show_padded_matrix(metrics)

x_array = dict2array(data.keys())
y_array = dict2array(data.values())

plt.plot(x_array, y_array, ls='none', marker='o', label='data')
plt.plot(x_array, avg, ls='--', label='avg.')
plt.plot(x_array, pred, label='lm')
plt.plot(x_array, test, label='test')
plt.plot(x_array, pred_d2, label='lm_d2')
plt.legend()

# justifying the 68, 95, 99.7 rule
std = np.std(y)
avg_ = np.mean(y)
rang = [avg_ - std, avg_ + std]
cond = (y > rang[0]) & (y < rang[1])
y_filt = y[cond]
y_filt.shape[0] / y.shape[0]

rang = [avg_ - 2 * std, avg_ + 2 * std]
cond = (y > rang[0]) & (y < rang[1])
y_filt = y[cond]
y_filt.shape[0] / y.shape[0]
md
#### Again but with feature scalling

# creating toy data
data = {x: 5 + x ** 2 for x in range(11)}
data = {k: v / 10 for k, v in data.items()}
test = [-0.5 + 0.12 * x for x in data.keys()]

df = pd.DataFrame(list(data.items()), columns=['x', 'y'])
X, y = df[['x']], df[['y']]

# feature scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)
avg = [X.mean() for i in range(X.shape[0])]

# fitting a linear model
lm = LinearRegression()
lm.fit(X, y)
pred = lm.predict(X)

X = pd.DataFrame(X, columns=['x'])
X['x2'] = X ** 2
lm.fit(X, y)
pred_d2 = lm.predict(X)

# at any given point the linear reg could be 11% of the values
# at any given point the random linear line could be around 30% of the values
preds = {'pred': pred, 'avg': avg, 'test': test, 'pred_d2': pred_d2}
metrics = [get_reg_metrics(y, v_pred, k_nm) for k_nm, v_pred in preds.items()]
show_padded_matrix(metrics)

plt.plot(X['x'], y, ls='none', marker='o', label='data')
plt.plot(X['x'], avg, ls='--', label='avg.')
plt.plot(X['x'], pred, label='lm')
plt.plot(X['x'], test, label='test')
plt.plot(X['x'], pred_d2, label='lm_d2')
plt.legend()

# justifying the 68, 95, 99.7 rule
std = np.std(y)
avg_ = np.mean(y)
rang = [avg_ - std, avg_ + std]
cond = (y > rang[0]) & (y < rang[1])
y_filt = y[cond]
y_filt.shape[0] / y.shape[0]

rang = [avg_ - 2 * std, avg_ + 2 * std]
cond = (y > rang[0]) & (y < rang[1])
y_filt = y[cond]
y_filt.shape[0] / y.shape[0]

md


### Confidence intervals and prediction intervals

class Relationship:
    """

    """

    def __init__(self, points=50):
        self.points = points
        self.x = np.arange(self.points).reshape(-1, 1)

    @property
    def y(self):
        y = 4 + self.x ** 2 + self.x ** 3 + self.x ** 4
        y = y / y.max()
        y = y + np.random.normal(0, .1, self.points).reshape(-1, 1) + 3
        return y.reshape(-1, 1)

    @property
    def x_const(self):
        return np.hstack((np.ones(50).reshape(-1, 1), self.x.reshape(-1, 1)))

    @property
    def x_poly(self):
        return np.hstack((self.x_const, self.x ** 2))

    @property
    def model(self):
        return sm.OLS(self.y, self.x_const).fit()

    @property
    def poly_model(self):
        return sm.OLS(self.y, self.x_poly).fit()

    @property
    def ci(self):
        st, data, ss2 = summary_table(self.model)
        mean_ci_low, mean_ci_upp = data[:, 4:6].T
        ci_low, ci_upp = data[:, 6:8].T
        return (mean_ci_low, mean_ci_upp, ci_low, ci_upp)

    @property
    def ci_poly(self):
        st, data, ss2 = summary_table(self.poly_model)
        mean_ci_low, mean_ci_upp = data[:, 4:6].T
        ci_low, ci_upp = data[:, 6:8].T
        return (mean_ci_low, mean_ci_upp, ci_low, ci_upp)

    def plot_xy(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.x, self.y, label='points')
        ax.plot(self.x, self.model.predict(self.x_const), label='linear')
        ax.plot(self.x, self.poly_model.predict(self.x_poly), label='poly')

        #         ax.plot(self.x, self.ci[0], label='mean_ci_low', c='green', ls='--')
        #         ax.plot(self.x, self.ci[1], label='mean_ci_upp', c='green', ls='--')
        #         ax.plot(self.x, self.ci[2], label='pred_ci_low', c='purple', ls='--')
        #         ax.plot(self.x, self.ci[3], label='pred_ci_upp', c='purple', ls='--')

        ax.plot(self.x, self.ci_poly[0], label='mean_ci_low', c='green', ls='--')
        ax.plot(self.x, self.ci_poly[1], label='mean_ci_upp', c='green', ls='--')
        ax.plot(self.x, self.ci_poly[2], label='pred_ci_low', c='purple', ls='--')
        ax.plot(self.x, self.ci_poly[3], label='pred_ci_upp', c='purple', ls='--')
        return (fig, ax)


from statsmodels.stats.outliers_influence import summary_table
from statsmodels.tools import eval_measures

relation = Relationship()

relation.plot_xy()

relation.model.summary()

relation.poly_model.summary()


mod = relation.poly_model

mod.conf_int()

mod.summary()
