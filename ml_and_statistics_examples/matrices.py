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

# matrix multiplication

a = np.array([
    [1, 2],
    [3, 4],
    [4, 3],
    [2, 1],
])
b = np.arange(1, 5).reshape(2, 2)
a.dot(b)
a @ b
