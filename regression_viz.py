"""
visualising regression surfaces of different models in 3d to help understand what a different models make decisions.

We have lower dimensional brains that do a good job of perseving 3d but terrible anything beyond that by visuzalising
surfaces in lower dimension it helps given you an intuative understanding what what it is doing in higher dimensions
which we cannot visualise.
"""

# importing modules
import time
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Union, Optional, Any, Sequence, Iterator
import logging
from datetime import datetime, timedelta
from pathlib import Path
from importlib import reload
import itertools
import functools
import io
import os
import gc
import re
import sys
import time
import logging
import pickle
import json
from my_helpers import slibtk, dptk, mltk, dviztk

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from functools import partial
import torch
import umap
import torch.nn as nn
from pygcp import pygcp
from pyflake import pyflake

import cufflinks as cf
import pandas as pd
import statsmodels.api as sm
import umap
from plotly.offline import init_notebook_mode, plot
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from my_helpers import mltk, dptk, slibtk
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / 'output'

# loading and preprocessing data #######################################################################################
boston = load_boston()
x = pd.DataFrame(boston['data'], columns=boston['feature_names'])
y = boston['target']
x['lbl'] = y

# normalising data
boston_norm = StandardScaler().fit_transform(x)
y = boston_norm[:, -1]
x_norm = boston_norm[:, :-1]

# pca
pcs, pca = mltk.pca_estimate(x_norm, n_components=2)
pcs = pd.DataFrame(data=pcs, columns=['dim0', 'dim1'])
pcs['lbl'] = y
pcs['type'] = 'pca'

# umap
reduced = umap.UMAP(n_components=2).fit_transform(x_norm)
reduced = pd.DataFrame(reduced, columns=[f'dim{i}' for i in range(2)])
reduced['lbl'] = y
reduced['type'] = 'umap'

# visulasing the difference between pca and umap
comp = pd.concat([pcs, reduced])
fig = px.scatter_3d(comp, 'dim0', 'dim1', 'lbl', color='type')
plot(fig)

# iterating over several models plotting them as 3d regression surfaces
n_trees = 300
params_xgb = {'colsample_bytree': 1.0,
              'gamma': 0.5,
              'max_depth': 3,
              'min_child_weight': 5,
              'subsample': 0.8}
params_lgbm = {'lambda_l1': 1.5,
               'lambda_l2': 1,
               'min_data_in_leaf': 30,
               'num_leaves': 31,
               'reg_alpha': 0.1}
models = {
    'Linear Regression': LinearRegression(),
    'SVM (Linear)': SVR(kernel='linear'),
    'SVM (rbf)': SVR(kernel='rbf', C=100, gamma=0.01),
    'Decision Tree': DecisionTreeRegressor(),
    f'Random Forest ({n_trees})': RandomForestRegressor(n_trees),
    'XGBoost': XGBRegressor(**params_xgb),
    'LightGBM': LGBMRegressor(**params_lgbm),
}

for nm, model in models.items():
    dviztk.px_scatter3d_regression(pcs, 'dim0', 'dim1', 'lbl', model, OUTPUT / f'{nm}.html', title=nm, marker_size=5)

# grid = GridSearchCV(LGBMRegressor(), mltk.param_lgbm)
# grid.fit(pcs.iloc[:, :3], pcs['lbl'])
# grid.best_params_

