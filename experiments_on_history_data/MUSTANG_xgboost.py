import pandas as pd
import numpy as np
import tqdm
import glob
import time
import pickle
import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor

X_train = pd.read_pickle('data/dataset_2/X_train.pickle')
X_test = pd.read_pickle('data/dataset_2/X_test.pickle')

# X_train_cats = pd.read_pickle('data/dataset_2/X_train_cats.pickle')
# X_test_cats = pd.read_pickle('data/dataset_2/X_test_cats.pickle')
y_train = pd.read_pickle('data/dataset_2/y_train.pickle')
y_test = pd.read_pickle('data/dataset_2/y_test.pickle')

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 500, 1000],
    'colsample_bytree': [0.3, 0.7]
}


xgb_model = XGBRegressor(enable_categorical=True, tree_method='hist')

grid_search = GridSearchCV(
    estimator=xgb_model, param_grid=param_grid, cv=5,
    scoring='neg_mean_absolute_error', verbose=1
)

grid_search.fit(X_train_cats, y_train)

print("Лучшие параметры:", grid_search.best_params_)

y_pred = grid_search.best_estimator_.predict(X_test_cats)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MSE на тестовой выборке:", mse)
print("MAPE на тестовой выборке:", mape)

best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}
xgb_model = XGBRegressor(enable_categorical=True, tree_method='hist', **best_params)
xgb_model.fit(X_train_cats, y_train)

y_pred = xgb_model.predict(X_test_cats)

mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

dct_imp = dict(zip(X_test_cats.columns, xgb_model.feature_importances_))
dct_imp = dict(sorted(dct_imp.items(), key=lambda s: s[1]))

features_names = list(dct_imp.keys())
features_imps = list(dct_imp.values())

plt.figure(figsize=(5, 7))
plt.barh(features_names, features_imps)
# plt.xticks(rotation=90)
plt.grid(True)
plt.ylabel('Признак', fontsize=12)
plt.xlabel('Важность признака', fontsize=12)
plt.show()

def figure_preds(y_test, y_pred):
    plt.figure(figsize=(10, 3))
    plt.plot(y_test, label='wait time')
    plt.plot(y_pred, label='predicted time')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, None])
    plt.xlim([0, len(y_test)])

print("MSE на тестовой выборке:", mse)
print("MAPE на тестовой выборке:", mape)
print("MAE на тестовой выборке:", mae)

figure_preds(y_test, y_pred)

xgb_model = XGBRegressor()

grid_search = GridSearchCV(
    estimator=xgb_model, param_grid=param_grid, cv=5,
    scoring='neg_mean_absolute_error', verbose=1
)

grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)

y_pred = grid_search.best_estimator_.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE на тестовой выборке:", mse)
print("MAPE на тестовой выборке:", mape)
print("MAE на тестовой выборке:", mae)
figure_preds(y_test, y_pred)

print(mse, mae, mape * 100)

preds_unscaled = target_scaler.inverse_transform(
    xgb_model.predict(X_test).reshape(-1, 1)
)
y_test_unscaled = target_scaler.inverse_transform(
    y_test.values.reshape(-1, 1)
)

mse = mean_squared_error(y_test_unscaled, preds_unscaled)
mape = mean_absolute_percentage_error(y_test_unscaled, preds_unscaled)
mae = mean_absolute_error(y_test_unscaled, preds_unscaled)

print(mse, mape, mae)

xgb_model = XGBRegressor(
    enable_categorical=True, tree_method='hist',
    **{'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500}
)

with open('data/version_2/target_scaler.pickle', 'rb') as f:
    target_scaler = pickle.load(f)

preds_unscaled = target_scaler.inverse_transform(
    xgb_model.predict(X_test_cats).reshape(-1, 1)
)
y_test_unscaled = target_scaler.inverse_transform(
    y_test.values.reshape(-1, 1)
)

mse = mean_squared_error(y_test_unscaled, preds_unscaled)
mape = mean_absolute_percentage_error(y_test_unscaled, preds_unscaled)
mae = mean_absolute_error(y_test_unscaled, preds_unscaled)

print(mse, mape, mae)