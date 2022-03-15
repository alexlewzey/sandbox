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
import string
import requests
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

from numpy import random

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.5f}'.format)

warnings.simplefilter(action='ignore')


# DISCRETE DISTRIBUTIONS ###############################################################################################

# binomial distribution ################################################################################################


def is_iterable(o: Any) -> bool:
    try:
        iter(o)
    except TypeError:
        return False
    else:
        return True


def plot_stats_dist(ns, ps, dist, nx=None, **kwargs):
    ns = ns if is_iterable(ns) else [ns]
    ps = ps if is_iterable(ps) else [ps]
    nx = nx if nx else max(ns)
    data = []
    for n in ns:
        for p in ps:
            x = np.arange(nx + 1)
            df = pd.Series(dist(n=n, p=p).pmf(x)).to_frame('pmf')
            df['n'] = n
            df['p'] = p
            df['x'] = x
            data.append(df)
    data = pd.concat(data)
    data['params'] = 'n=' + data['n'].astype(str) + ', p=' + data['p'].astype(str)
    px.line(data, 'x', 'pmf', color='params', **kwargs).plot()


def plot_binomial(ns: Optional[Any] = 50, ps: Optional[Any] = 0.5, **kwargs):
    plot_stats_dist(ns, ps, stats.binom, **kwargs)


plot_binomial(ps=np.linspace(0, 1, 8))
plot_binomial(ns=range(1, 50, 10))
plot_binomial(ns=range(1, 50, 10), ps=0.7)

# cdf is equal to the sum of the pmf to the left side of the distribution
stats.binom(n=10, p=0.5).cdf(5), stats.binom(n=10, p=0.5).pmf(range(6)).sum()

dist = stats.binom(n=1000, p=0.5)
x = np.arange(300, 700)
df = pd.DataFrame({
    'x': x,
    'pmf': dist.pmf(x),
    'cdf': dist.cdf(x),
})
dviztk.line_seconday_axis(df, 'x', 'pmf', 'cdf').plot()
p_value = 1 - dist.cdf(550)
if p_value < 0.01:
    print(f'reject the null hypothesis that the coin unbaised in favor of the alternative hypothesis')

# plotting samples
b = stats.binom(n=100, p=0.7)
b_data = pd.Series([b.rvs() for _ in range(100_000)]).to_frame('binom')
b_data['binom'].px_hist()

# a coin is flipped 1000 times and returns heads 550 times, is it baised?
dist = stats.binom(n=1000, p=0.5)
# you can argue that this is a 2 sided test
p_values = 1 - dist.cdf(550) + dist.cdf(450)


# geometric ############################################################################################################

def geometric(x, p):
    return ((1 - p) ** (x - 1)) * p


dist = stats.geom(p=0.5)
assert geometric(1, 0.5) == dist.pmf(1)

# plotting the geometric pmf
data = []
for mu in np.linspace(0, 1, 5):
    dist = stats.geom(p=mu)
    x = np.arange(10)
    pmf = dist.pmf(x)
    df = pd.DataFrame({'x': x, 'pmf': pmf})
    df['p'] = mu
    data.append(df)
data = pd.concat(data)
px.line(data, 'x', 'pmf', color='p').plot()

# how many attempts would indicate an unfair coin
dist = stats.geom(p=0.5)
# there is only a 0.008% chance of requiring 7 tosses is you are using a fair coin i.e. significant at the
# 1% confidence level
p_value = 1 - dist.cdf(7)
print(p_value)

# negative binomial ####################################################################################################

# the distribution starts from the nth trial i.e. x is the number of failures required to achieve n successes
dist = stats.nbinom(p=0.5, n=2)
x = np.arange(0, 50)
pmf = dist.pmf(x)
df = pd.DataFrame({'x': x, 'pmf': pmf})
px.line(df, 'x', 'pmf').plot()

plot_stats_dist(ns=range(1, 50, 10), ps=0.5, dist=stats.nbinom, nx=100, title='by no. of required successes')
plot_stats_dist(ns=10, ps=np.linspace(0, 1, 5), dist=stats.nbinom, nx=100, title='by probability of success in the bernoilli trials')

# poisson ##############################################################################################################


# plotting the geometric pmf
data = []
for mu in np.linspace(3, 100, 5):
    dist = stats.poisson(mu=mu)
    x = np.arange(150)
    pmf = dist.pmf(x)
    df = pd.DataFrame({'x': x, 'pmf': pmf})
    df['mu'] = mu
    data.append(df)
data = pd.concat(data)
px.line(data, 'x', 'pmf', color='mu', title='poission dist by lambda').plot()

# zed statistics problem
dist = stats.poisson(mu=12)
pmf = dist.pmf(10)
cdf = (1 - dist.cdf(9))
print(f'pmf={pmf:.3f}, cdf={cdf:.3f}')
dist = stats.poisson(mu=12 / 24)
1 - dist.cdf(1)

# hypergeometric #######################################################################################################

# normal - scipy
# N = M
# A = n
# n = N
# x = k

# zed statistics question
dist = stats.hypergeom(M=50, n=11, N=5)
1 - dist.cdf(2)

# CONTINUOUS DISTRIBUTIONS #############################################################################################


# normal / gaussian distribution #######################################################################################

data = []
for loc in [0, 20]:
    for scale in [1, 5]:
        x = np.linspace(-15, 33, 1000)
        df = pd.Series(stats.norm(loc=loc, scale=scale).pdf(x)).to_frame('pdf')
        df['loc'] = loc
        df['scale'] = scale
        df['x'] = x
        data.append(df)
data = pd.concat(data)
data['params'] = 'loc=' + data['loc'].astype(str) + ', scale=' + data['scale'].astype(str)
px.line(data, 'x', 'pdf', color='params', title='normal distribution by parameters').plot()

# plotting a multivariate normal distribution
dist = np.random.randn(1000, 2)
fig, ax = plt.subplots()
ax.scatter(dist[:, 0], dist[:, 1] * 2 + 1, alpha=0.3)
ax.axis('equal')
fig.show()

# exponential ##########################################################################################################


# zed statistics questions
scale = 1 / (3 / 60)
dist = stats.expon(scale=scale)
a = dist.cdf(10)
b = 1 - dist.cdf(30)
c = dist.pdf(15)
print(f'a={a:.3f}, b={b:.3f}, c={c:.3f}')

# beta #################################################################################################################
# the probability of a probability

dist = stats.beta(a=3, b=2)
df = pd.Series(dist.pdf(np.linspace(0, 1, 100))).to_frame('a').reset_index()
px.line(df, 'index', 'a').plot()


def beta_plots():
    data = []
    for a, b in [
        (3, 7),
        (30, 70),
        (300, 700),
        (6, 4),
        (60, 40),
        (600, 400),
    ]:
        x = np.linspace(0, 1, 1000)
        df = pd.Series(stats.beta(a=a, b=b).pdf(x)).to_frame('pdf')
        df['a'] = a
        df['b'] = b
        df['x'] = x
        data.append(df)
    data = pd.concat(data)
    data['params'] = 'a=' + data['a'].astype(str) + ', b=' + data['b'].astype(str)
    px.line(data, 'x', 'pdf', color='params', title='beta distribution by parameters').plot()


beta_plots()

# you roll and get 10 head and 16 tails, what is the probablity this is a fair coin
obs = [8, 18]
obs2 = [i * 3 for i in obs]
n_trials = sum(obs)
n_heads = obs[0]

# binomial solution
dist = stats.binom(n=sum(obs), p=0.5)
df = pd.Series(dist.pmf(np.arange(dist.ppf(0), dist.ppf(1)))).to_frame('pmf').reset_index()
px.bar(df, 'index', 'pmf').plot()
p_value = dist.cdf(n_heads) * 2
print(p_value)

# with a larger sample size the coin is definatly baised
dist = stats.binom(n=sum(obs2), p=0.5)
df = pd.Series(dist.pmf(np.arange(dist.ppf(0), dist.ppf(1)))).to_frame('pmf').reset_index()
px.bar(df, 'index', 'pmf').plot()
p_value = dist.cdf(obs2[0]) * 2
print(p_value)

# chi-squared method
print(stats.chisquare(obs), '\n', stats.chisquare(obs2))

# beta distribution approach
dist = stats.beta(a=obs[0], b=obs[1])
x = np.linspace(dist.ppf(0), dist.ppf(1), 1000)
df = pd.Series(dist.pdf(x)).to_frame('pdf').reset_index()
df['x'] = x
px.bar(df, 'x', 'pdf').plot()

max_pdf_prob = (df['pdf'].argmax() + 1) / df.shape[0]
prob_heads = np.array(obs) / sum(obs)
print(f'max_pdf_prob={max_pdf_prob:.3f}, prob_heads={prob_heads}')
p_value = 1 - dist.cdf(0.5)
p_value

# multivariate distribution ############################################################################################


cov = np.array([
    [1, 2],
    [2, 2],
])

mean = np.array([0, 2])

multivar_randn = stats.multivariate_normal.rvs(mean=mean, cov=cov, size=1000)

fig, ax = plt.subplots()
ax.scatter(multivar_randn[:, 0], multivar_randn[:, 1] * 2 + 1, alpha=0.3)
ax.axis('equal')
fig.show()

stats.binom.ppf(0.95, n=10, p=0.9)
