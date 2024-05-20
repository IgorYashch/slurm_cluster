import pandas as pd
import numpy as np
import tqdm
import glob
import time
import pickle
import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from scipy.stats import gaussian_kde


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

X_train = pd.read_pickle('data/dataset_2/X_train.pickle')
X_test = pd.read_pickle('data/dataset_2/X_test.pickle')
y_train = pd.read_pickle('data/dataset_2/y_train.pickle')
y_test = pd.read_pickle('data/dataset_2/y_test.pickle')

with open('data/version_2/target_scaler.pickle', 'rb') as f:
    target_scaler = pickle.load(f)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge 5": Ridge(alpha=5),
    "Ridge 15": Ridge(alpha=15),
    "Random Forest": RandomForestRegressor(),
    "Random Forest 4": RandomForestRegressor(max_depth=4),
#     "Random Forest MAE": RandomForestRegressor(criterion='absolute_error'),
    "Gradient Boosting": GradientBoostingRegressor(),
    "LGBM": LGBMRegressor(),
    "LGBM dart": LGBMRegressor(boosting_type='dart'),
    "LGBM 4": LGBMRegressor(max_depth=4)
}

def count_results(models, X_train, y_train, X_test, y_test, target_scaler):

    results = pd.DataFrame(
        index=['MSE train', 'MAE train', 'MAPE train', 'MSE test', 'MAE test', 'MAPE test', 'time'],
        columns=list(models)
    )

    results_unscaled = pd.DataFrame(
        index=['MSE train', 'MAE train', 'MAPE train', 'MSE test', 'MAE test', 'MAPE test', 'time'],
        columns=list(models)
    )
    
    y_train_i = target_scaler.inverse_transform(y_train.values.reshape(-1, 1))
    y_test_i = target_scaler.inverse_transform(y_test.values.reshape(-1, 1))
    
    y_preds_dict = dict()
        
    for model_name in tqdm.tqdm(models):
        
        print(model_name)
        
        model = models[model_name]
        t_start = time.time()
        model = model.fit(X_train, y_train)
        y_train_hat = model.predict(X_train)
        y_test_hat = model.predict(X_test)
        t_finish = time.time()
        
        y_train_hat_i = target_scaler.inverse_transform(y_train_hat.reshape(-1, 1))
        y_test_hat_i = target_scaler.inverse_transform(y_test_hat.reshape(-1, 1))

        mse_train = mean_squared_error(y_train, y_train_hat)
        mae_train = mean_absolute_error(y_train, y_train_hat)
        mape_train = mean_absolute_percentage_error(y_train, y_train_hat)

        mse = mean_squared_error(y_test, y_test_hat)
        mae = mean_absolute_error(y_test, y_test_hat)
        mape = mean_absolute_percentage_error(y_test, y_test_hat)
        
        y_preds_dict[model_name + '_test'] = y_test_hat
        y_preds_dict[model_name + '_train'] = y_train_hat

        results.loc['MSE train', model_name] = mse_train
        results.loc['MAE train', model_name] = mae_train
        results.loc['MAPE train', model_name] = mape_train
        results.loc['MSE test', model_name] = mse
        results.loc['MAE test', model_name] = mae
        results.loc['MAPE test', model_name] = mape
        results.loc['time', model_name] = t_finish - t_start
        
        
        mse_train = mean_squared_error(y_train_i, y_train_hat_i)
        mae_train = mean_absolute_error(y_train_i, y_train_hat_i)
        mape_train = mean_absolute_percentage_error(y_train_i, y_train_hat_i)

        mse = mean_squared_error(y_test_i, y_test_hat_i)
        mae = mean_absolute_error(y_test_i, y_test_hat_i)
        mape = mean_absolute_percentage_error(y_test_i, y_test_hat_i)
        
        y_preds_dict[model_name + '_test_unsc'] = y_test_hat_i
        y_preds_dict[model_name + '_train_unsc'] = y_train_hat_i
        
        results_unscaled.loc['MSE train', model_name] = mse_train
        results_unscaled.loc['MAE train', model_name] = mae_train
        results_unscaled.loc['MAPE train', model_name] = mape_train
        results_unscaled.loc['MSE test', model_name] = mse
        results_unscaled.loc['MAE test', model_name] = mae
        results_unscaled.loc['MAPE test', model_name] = mape
        results_unscaled.loc['time', model_name] = t_finish - t_start
    
    return results, results_unscaled, y_preds_dict

results, results_unscaled, y_preds_dict = count_results(models, X_train, y_train, X_test, y_test, target_scaler)

y_test_i = target_scaler.inverse_transform(y_test.values.reshape(-1, 1))

test_diff = pd.DataFrame(y_test_i - y_preds_dict['Linear Regression_test_unsc'])

exp_num = 2

results.to_csv(f'exp_dataset_2/exp_{exp_num}/results_scaled.csv')
results_unscaled.to_csv(f'exp_dataset_2/exp_{exp_num}/results_unscaled.csv')

X_train.to_csv(f'exp_dataset_2/exp_{exp_num}/X_train.csv.gz', compression='gzip')
y_train.to_csv(f'exp_dataset_2/exp_{exp_num}/y_train.csv.gz', compression='gzip')
X_test.to_csv(f'exp_dataset_2/exp_{exp_num}/X_test.csv.gz', compression='gzip')
y_test.to_csv(f'exp_dataset_2/exp_{exp_num}/y_test.csv.gz', compression='gzip')


with open(f'exp_dataset_2/exp_{exp_num}/y_scaler.pickle', 'wb') as f:
    pickle.dump(target_scaler, f)
    
for model_name in models:
    new_name = model_name.replace(' ', '_')
    with open(f'exp_dataset_2/exp_{exp_num}/{new_name}.pickle', 'wb') as f:
        pickle.dump(models[model_name], f)


results.T[['MSE train','MSE test', 'MAE train', 'MAE test', 'MAPE train',
       'MAPE test', 'time']]


res_cold = results.T[['MSE train','MSE test', 'MAE train', 'MAE test', 'MAPE train',
       'MAPE test', 'time']]


res_cold[res_cold == res_cold.min()]

res_cold[['MSE test', 'MAE test', 'MAPE test', 'time']]

results_unscaled.T[['MSE test', 'MAE test', 'MAPE test', 'time']]


results_unscaled.T[['MSE train','MSE test', 'MAE train', 'MAE test', 'MAPE train',
       'MAPE test', 'time']]


results_unscaled = pd.read_csv('results_classical_ml/results_unscaled.csv', index_col=0)[['MSE test', 'MAE test', 
       'MAPE test', 'time']]


results_scaled = pd.read_csv('results_classical_ml/results.csv', index_col=0)[['MSE test', 'MAE test', 
       'MAPE test', 'time']]

def figure_preds(y_test, y_pred):
    plt.figure(figsize=(10, 3))
    plt.plot(y_test, label='wait time')
    plt.plot(y_pred, label='predicted time')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, len(y_test)])
    
def figure_preds_real(y_test, y_pred):
    y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1))
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    plt.figure(figsize=(10, 3))
    plt.plot(y_test, label='wait time')
    plt.plot(y_pred, label='predicted time')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, len(y_test)])

with open(f'exp_dataset_2/exp_1/Random_Forest.pickle', 'rb') as f:
    model = pickle.load(f)
    
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)


figure_preds_real(y_test, test_preds)


with open(f'exp_dataset_2/exp_1/LGBM.pickle', 'rb') as f:
    model = pickle.load(f)
    
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

figure_preds(y_test, test_preds)