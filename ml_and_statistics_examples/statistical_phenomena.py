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


# central limit theorm #################################################################################################

def dist_rug(a, ax):
    sns.distplot(a, ax=ax, norm_hist=True)
    sns.rugplot(a, ax=ax)

def get_sample_means(a, n_sample, size_sample):
    means = []
    for i in range(n_sample):
        sample = np.random.choice(a, size=size_sample)
        means.append(np.mean(sample))
    return means

exp = np.random.exponential(size=100)
means = get_sample_means(a=exp, n_sample=50, size_sample=15)

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
dist_rug(exp, ax)
dist_rug(means, ax)

uni = np.random.uniform(size=100)
means = get_sample_means(uni, 30, 10)
fig, ax = plt.subplots(1, 1, figsize=(9, 4))
dist_rug(uni, ax)
dist_rug(means, ax)

# the law of large numbers #############################################################################################

# As n increases the probability distribution of the sample converges to the probability distribution of the populaiton

# flipping a coin
observed_vals = {}
n = 1000
for i in range(1, n):
    tosses = np.random.randint(0, 2, size=i)
    exp_val = np.mean(tosses)
    observed_vals[i] = exp_val

observed_x, observed_y = dict2array(observed_vals.keys()), dict2array(observed_vals.values())
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(observed_x, observed_y)
ax.set(ylabel='avg value', xlabel='no. coin tosses', title='law of large numbers')
ax.axhline(0.5, ls='--', c='orange')
plt.plot()