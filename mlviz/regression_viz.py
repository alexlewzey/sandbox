"""
visualising regression surfaces of different models in 3d to help understand what a different models make decisions.

We have lower dimensional brains that do a good job of perseving 3d but terrible anything beyond that by visuzalising
surfaces in lower dimension it helps given you an intuative understanding what what it is doing in higher dimensions
which we cannot visualise.
"""

import logging
# importing modules
from pathlib import Path

import pandas as pd
import plotly.express as px
import umap
from lightgbm import LGBMRegressor
from plotly.offline import plot
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from dstk import mltk, dviztk

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / 'mlviz' / 'output'

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
res = mltk.pca(x_norm, n_components=2)
pcs, pca = res['pcs'], res['pca']
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
plot(fig, OUTPUT / 'pca_vs_umap.html')

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
