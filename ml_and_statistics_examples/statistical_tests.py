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
from scipy.stats import chisquare, chi2
from google.cloud import storage, bigquery, secretmanager
import warnings

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.5f}'.format)

warnings.simplefilter(action='ignore')

# Statistical tests:
#  - chi squared
#  - logistic regression
#  - t-test
#  - ANOVA
#  - linear regression
#  - multivariate regression


# t-test ###############################################################################################################


N = 500
a = np.random.randn(100)
b = np.random.randn(120) + 0.5

fig, ax = plt.subplots(1, 1, figsize=(9, 3.5))
ax.hist(a, alpha=0.5)
ax.hist(b, alpha=0.5)
plt.show()

# 1 sample t-test ######################################################################################################

for i in [0.5, 0.6, 0.7, 0.8]:
    t_value, p_value = stats.ttest_1samp(b, i)
    print(f't_value={t_value:.2f}, p_value={p_value:.3f}, mean={i:.1f}')
    if p_value < 0.05:
        print(f'we reject the null hypothesis that the mean is equal to {i}')
    else:
        print(f'we fail to reject the null hypothesis that there is no difference between the means')

# independent t-test ####################################################################################################

# using libraries

t_func, pv = stats.ttest_ind(a, b, equal_var=False)
print(t_func, pv)

# from scratch
# cumulative distribution function (CDF)
# percent point function (PPF)

a_mn, b_mn = np.mean(a), np.mean(b)
signal = a_mn - b_mn

a_var, b_var = a.var(ddof=1), b.var(ddof=1)
a_n, b_n = len(a), len(b)
noise = np.sqrt((a_var / a_n) + (b_var / b_n))

observed = (a_n + b_n) - 2
t = signal / noise
p = stats.t.cdf(t, df=observed) * 2

print(t, p)


def independent_ttest(data1, data2, alpha):
    mean1, mean2 = np.mean(data1), np.mean(data2)
    se1, se2 = stats.sem(data1), stats.sem(data2)
    sed = np.sqrt(se1 ** 2.0 + se2 ** 2.0)
    t_stat = (mean1 - mean2) / sed
    df = len(data1) + len(data2) - 2
    cv = stats.t.ppf(1.0 - alpha, df)
    p = (1.0 - stats.t.cdf(abs(t_stat), df)) * 2.0
    return t_stat, df, cv, p


t_stat, observed, cv, p = independent_ttest(a, b, 0.05)
print(t_stat, p)

# paired (dependent) t-test ############################################################################################

a = np.random.randn(100)
b = np.random.randn(100) + 1
c = np.random.randn(100)

t_value, p_value = stats.ttest_rel(a, b)
print(t_value, p_value)
_value, p_value = stats.ttest_rel(a, c)
print(t_value, p_value)


# chi squared test #####################################################################################################

# from scratch

def chi(observed, expected):
    return sum((observed - expected) ** 2 / expected)


o = np.array([8.9, 1.1])
e = np.array([5, 5])
chi(o, e)

# from packages

o = np.array([30, 14, 34, 45, 57, 20])
e = np.array([20, 20, 30, 40, 60, 30])

observed = len(o) - 1
chi_val_5pct = stats.chi2.isf(q=0.05, df=observed)
chi_val, p_value = stats.chisquare(f_obs=o, f_exp=e)
print(f'chi_val_5pct={chi_val_5pct:.3f}, chi_val={chi_val:.3f}, p_value={p_value:.3f}')
if p_value < 0.05:
    print(f'we reject the null hypothesis that these two samples come from the same distribution')
else:
    print(f'we fail to reject the null hypothesis')

# coin toss example

o = [4, 0]
e = [2, 2]
chi, p_value = stats.chisquare(o)

o = [5, 0]
chi, p_value = stats.chisquare(o)

# zedstatistics

o = [75 - 11, 11]
e_dist = [1 - 0.12, 0.12]
e = [i * sum(o) for i in e_dist]
chi, p_value = stats.chisquare(o, e)
print(f'chi={chi:.3f}, p_value={p_value:.3f}')

o = [235, 194, 171]
e = [200, 200, 200]
chi, p_value = stats.chisquare(o, e)
print(f'chi={chi:.3f}, p_value={p_value:.3f}')

observed = pd.DataFrame({
    'male': {
        'facebook': 15,
        'instagram': 30,
        'tik tok': 5,
    },
    'female': {
        'facebook': 20,
        'instagram': 35,
        'tik tok': 15,
    },
})

expected = observed.copy()
total = observed.sum().sum()
for i in observed.index:
    for c in observed.columns:
        expected.loc[i, c] = (observed.loc[i].sum() * observed[c].sum()) / total
dof = np.product(np.array(observed.shape) - 1)
chi, p_value = stats.chisquare(observed.values.ravel(), expected.values.ravel(), ddof=dof)
print(f'chi={chi:.3f}, p_value={p_value:.3f}, dof={dof}, expected_=\n{expected:}')

chi_, p_value_, dof_, expected_ = stats.chi2_contingency(observed)
print(f'chi_={chi_:.3f}, p_value_={p_value_:.3f}, dof_={dof_}, expected_=\n{expected_:}')

# non-parametric tests #################################################################################################


# stats.mannwhitneyu()
# stats.wilcoxon()
