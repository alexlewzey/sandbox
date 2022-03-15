import math
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
import math
from google.cloud import storage, bigquery, secretmanager
import warnings

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.5f}'.format)

warnings.simplefilter(action='ignore')

x = list(np.arange(6) + 1)


# permutaitons #########################################################################################################
# order matters

def permutation(n, k):
    return math.factorial(n) / math.factorial(n - k)


p = list(itertools.permutations(x))
assert len(p) == math.factorial(6)

n, k = 6, 2
p = list(itertools.permutations(x, r=k))
f = math.factorial(n) / math.factorial(n - k)
assert len(p) == f


def permutation_with_repitition(n, k):
    return n ** k


p = list(itertools.product(x, repeat=n))
len(p), n ** n

p = list(itertools.product(x, repeat=k))
len(p), n ** k


# combinations #########################################################################################################
# order does not matter

def combination(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


# if n == k then there is only one combination
p = list(itertools.combinations(x, r=len(x)))

p = list(itertools.combinations(x, r=2))
assert combination(len(x), 2) == len(p)


# combinations with replacement

def combination_with_repetition(n, k):
    return math.factorial(k + n - 1) / (math.factorial(k) * math.factorial(n - 1))


p = list(itertools.combinations_with_replacement(x, r=k))
assert combination_with_repetition(n, k) == len(p)


# probability examples #################################################################################################


p = list(itertools.product(x, repeat=2))
eq4 = [i for i in p if sum(i) == 4]
print(f'p: len={len(p)}, eq4={len(eq4)}')



