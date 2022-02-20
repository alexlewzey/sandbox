import logging
from typing import *

import lightgbm as lgbm
import numpy as np
import pandas as pd
import plotly.express as px
from dstk import mltk
from sklearn import model_selection, metrics, datasets
from slibtk import slibtk

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.5f}'.format)


def eval_regression(df: pd.DataFrame, y: str = 'y', y_pred: str = 'y_pred') -> Tuple[pd.DataFrame, Dict]:
    scores = {
        'rmse': np.sqrt(metrics.mean_squared_error(df[y], df[y_pred])),
        'r2': metrics.r2_score(df[y], df[y_pred]),
        'std': df[y].std(),
        'iqr': df[y_pred].quantile(0.75) - df[y_pred].quantile(0.25),
    }
    scores = {k: f'{v:.2f}' for k, v in scores.items()}
    print(scores)
    df['residuals'] = (df[y_pred] - df[y]) + df[y].mean()
    px.scatter(df, y, y_pred, title='residuals').plot()
    px.histogram(df[[y, 'residuals']].melt(), 'value', color='variable', opacity=0.6, barmode='overlay', title='residuals', marginal='box').plot()
    return df, scores


def make_regression() -> Tuple[pd.DataFrame, List]:
    x, y = datasets.make_regression(100_000, 200, noise=0.30, n_informative=20)
    x = np.hstack([x, y.reshape(-1, 1)])
    features = [f'x{i}' for i in range(x.shape[1] - 1)]
    x = pd.DataFrame(x, columns=features + ['y'])
    return x, features


# keep everything in the same DataFrame and select columns where required


x, features = make_regression()
scaler, x['y_norm'] = mltk.min_max_scale(x['y'])
train, test = model_selection.train_test_split(x, test_size=0.2)

log = slibtk.LogTXT('results.txt')
log.clear()

# normal data ##########################################################################################################
model = lgbm.LGBMRegressor()
model.fit(train[features], train['y'])
test['y_pred'] = model.predict(test[features])
test, scores = eval_regression(test)
log.write(f'normal: {scores}')
model = lgbm.LGBMRegressor()
model.fit(train[features], train['y_norm'])
test['y_pred'] = model.predict(test[features])
test, scores = eval_regression(test, y='y_norm')
log.write(f'normal_norm: {scores}')

# reduced data #########################################################################################################

x_rd, pca = mltk.pca(x[features])
features_rd = x_rd.columns
x_rd = x_rd.join(x['y_norm']).join(x['y'])
train, test = model_selection.train_test_split(x_rd, test_size=0.2)

model = lgbm.LGBMRegressor()
model.fit(train[features_rd], train['y'])
test['y_pred'] = model.predict(test[features_rd])
test, scores = eval_regression(test, y='y')
log.write(f'reduced_norm: {scores}')
model = lgbm.LGBMRegressor()
model.fit(train[features_rd], train['y_norm'])
test['y_pred'] = model.predict(test[features_rd])
test, scores = eval_regression(test, y='y_norm')
log.write(f'reduced_norm: {scores}')

# reduced with dropped columns #########################################################################################

model = lgbm.LGBMRegressor()
model.fit(train[features_rd[:30]], train['y'])
test['y_pred'] = model.predict(test[features_rd[:30]])
test, scores = eval_regression(test, y='y')
log.write(f'reduced_dropped_columns: {scores}')
model = lgbm.LGBMRegressor()
model.fit(train[features_rd[:30]], train['y_norm'])
test['y_pred'] = model.predict(test[features_rd[:30]])
test, scores = eval_regression(test, y='y_norm')
log.write(f'reduced_dropped_columns_norm: {scores}')

# random data ##########################################################################################################

train_rand = np.random.rand(*train[features].shape)
test_rand = np.random.rand(*test[features].shape)

model = lgbm.LGBMRegressor()
model.fit(train_rand, train['y'])
test['y_pred'] = model.predict(test_rand)
test, scores = eval_regression(test)
log.write(f'random: {scores}')
model = lgbm.LGBMRegressor()
model.fit(train_rand, train['y_norm'])
test['y_pred'] = model.predict(test_rand)
test, scores = eval_regression(test, y='y_norm')
log.write(f'random_norm: {scores}')
