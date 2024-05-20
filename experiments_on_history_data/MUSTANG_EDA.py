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


scheduler_data = pd.read_csv('./data/mustang_release_v1.0beta.csv.gz')
scheduler_data.columns


scheduler_data.columns = ['user_ID', 'group_ID', 'time_submit', 'time_start', 'time_end', 'wallclock_limit', 'job_status', 'node_count', 'tasks_requested']


full = scheduler_data
scheduler_data = full[:500000]

scheduler_data.isna().sum()


print(scheduler_data[scheduler_data.time_submit.isna()].iloc[0])
scheduler_data[scheduler_data.time_submit.isna()].describe()

print(scheduler_data[scheduler_data.time_start.isna()].iloc[18])
scheduler_data[scheduler_data.time_start.isna()].describe()


scheduler_data = scheduler_data[~scheduler_data.time_submit.isna()]

scheduler_data["time_start"] = scheduler_data["time_start"].fillna(scheduler_data["time_submit"])


scheduler_data.isna().sum()


for col in ['time_submit', 'time_start', 'time_end']:
    scheduler_data[col] = pd.to_datetime(scheduler_data[col]).apply(lambda x: x.timestamp())

def convert_to_seconds(time_str):
    days_part, time_part = time_str.split(' days ')
    hh, mm, ss = time_part.split(':')
    total_seconds = int(int(days_part) * 86400 + int(hh) * 3600 + int(mm) * 60 + float(ss))
    return total_seconds


scheduler_data['wallclock_limit'] = scheduler_data['wallclock_limit'].apply(convert_to_seconds).apply(lambda x: x if x <  57600 else 0)

scheduler_data['time_wait'] = scheduler_data['time_start'] - scheduler_data['time_submit']

full_data = scheduler_data
scheduler_data = scheduler_data.iloc[:300000]

wait_time_train, wait_time_test = train_test_split(
    scheduler_data.reset_index()['time_wait'],
    shuffle=False
)
train_index = wait_time_train.index
test_index = wait_time_test.index

scheduler_data.head()
scheduler_data

plt.figure(figsize=(12, 4))
plt.plot(train_index, wait_time_train, label='Обучающая выборка')
plt.plot(test_index, wait_time_test, label='Тестовая выборка')
plt.legend()
plt.grid(True)
plt.ylim([0, None])
plt.xlim([0, len(scheduler_data)])
plt.ylabel('Время ожидания, с', fontsize=12)
plt.xlabel('Индекс задачи', fontsize=12)
plt.savefig('pictures_for_diploma/mustang_wait_time_plot.png')


max_waittime = max(wait_time_train.max(), wait_time_test.max())


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

hist_param = {
    'edgecolor': 'black', 'bins': np.linspace(0, max_waittime, 31)
}

axes[0].hist(wait_time_train, label='Обучающая выборка', **hist_param)
axes[1].hist(wait_time_test, label='Тестовая выборка', **hist_param, color='C1')

for ax in axes:
    ax.set_yscale('log')
    ax.grid(True)
    ax.set_xlabel('Время ожидания, ч', fontsize=12)
    ax.set_ylabel('Количество задач', fontsize=12)
    ax.legend()

plt.tight_layout()
plt.savefig('pictures_for_diploma/wait_time_distr.png')

def make_new_features(scheduler_time_data, row):
    new_features_dict = {}
    
    mask_1 = scheduler_time_data.time_submit < row["time_submit"]
    mask_2 = scheduler_time_data.time_start > row["time_submit"]
    mask_3 = scheduler_time_data.time_start < row["time_submit"]
    mask_4 = scheduler_time_data.time_end > row["time_submit"]

    query = scheduler_time_data[mask_1 & mask_2].copy()
    running = scheduler_time_data[mask_3 & mask_4].copy()

    query['now_time_wait'] = row["time_submit"] - query["time_submit"]
    running['now_time_run'] = row["time_submit"] - running["time_start"]
    
    for col in [running['time_wait'], running['now_time_run'], query['now_time_wait']]:
        new_features_dict[col.name + '_mean'] = col.mean()
        new_features_dict[col.name + '_median'] = col.median()
        new_features_dict[col.name + '_max'] = col.max()
        
    new_features_dict['wait_count'] = len(query)
    new_features_dict['run_count'] = len(running)
    
    ################################################################################################
    
    new_features_dict['run_nodes_sum'] = running["node_count"].sum()
    new_features_dict['wait_tasks_sum'] = query["tasks_requested"].sum()

    new_features_dict['run_user_count'] = len(running[running["user_ID"] == row["user_ID"]])
    new_features_dict['run_group_count'] = len(running[running["group_ID"] == row["group_ID"]])
    new_features_dict['query_user_count'] = len(query[query["user_ID"] == row["user_ID"]])
    new_features_dict['query_group_count'] = len(query[query["group_ID"] == row["group_ID"]])

    new_features_dict['run_user_nodes'] = running[running["user_ID"] == row["user_ID"]].node_count.sum()
    new_features_dict['run_group_nodes'] = running[running["group_ID"] == row["group_ID"]].node_count.sum()
    new_features_dict['query_user_req_tasks'] = query[query["user_ID"] == row["user_ID"]].tasks_requested.sum()
    new_features_dict['query_group_req_tasks'] = query[query["group_ID"] == row["group_ID"]].tasks_requested.sum()


    # last_submited = scheduler_time_data[mask_1].reset_index(drop=True)[-10:]
    # last_submited['now_time_wait'] = row["time_submit"] - last_submited.time_submit
    # last_submited['real_wait_time'] = [min(a, b) for a, b in zip(
    #     last_submited['now_time_wait'], last_submited['time_wait']
    # )]

    # new_features_dict['last_5_wait_time'] = last_submited[-5:].real_wait_time.median()
    # new_features_dict['last_10_wait_time'] = last_submited[-10:].real_wait_time.median()
    
    return new_features_dict


from joblib import Parallel, delayed
import pandas as pd

dataset_f = Parallel(n_jobs=-1, verbose=1)(
    delayed(make_new_features)(scheduler_data, row)
    for i, row in scheduler_data.iterrows()
)

dataset_f = pd.DataFrame(dataset_f).fillna(0)
dataset_f['wallclock_limit'] = scheduler_data['wallclock_limit'].values
dataset_f['tasks_requested'] = scheduler_data['tasks_requested'].values

scheduler_data.wallclock_limit.isna().sum(), scheduler_data.tasks_requested.isna().sum()#, scheduler_data.cpus_req.isna().sum()

y_vector = scheduler_data['time_wait']

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].hist(
    dataset_f.wait_count[dataset_f.wait_count < 6000],
    bins=40, color='C0', edgecolor='black'
)
axs[0].grid(True)
axs[0].set_xlabel('Количество задач в очереди', fontsize=12)
axs[0].set_ylabel('Количество задач', fontsize=12)

axs[1].hist(dataset_f.run_count, bins=40, color='C0', edgecolor='black')
axs[1].grid(True)
axs[1].set_xlabel('Количество исполняющихся задач', fontsize=12)
axs[1].set_ylabel('Количество задач', fontsize=12)

plt.tight_layout()
plt.show()
plt.savefig('pictures_for_diploma/num_tasks_wait_run.png')

fig, axs = plt.subplots(3, 3, figsize=(12, 10))

columns = ['time_wait_mean', 'now_time_run_mean', 'now_time_wait_mean', 
           'time_wait_median', 'now_time_run_median', 'now_time_wait_median', 
           'time_wait_max', 'now_time_run_max', 'now_time_wait_max']
titles = ['Среднее время ожидания (задачи в выполнении)', 'Среднее время выполнения (задачи в выполнении)', 'Среднее время ожидания (задачи в очереди)', 
          'Медианное время ожидания (задачи в выполнении)', 'Медианное время выполнения (задачи в выполнении)',  'Медианное время ожидания (задачи в очереди)',
          'Максимальное время ожидания (задачи в выполнении)', 'Максимальное время выполнения (задачи в выполнении)',  'Максимальное время ожидания (задачи в очереди)']

for ax, column, title in zip(axs.flat, columns, titles):
    ax.hist(dataset_f[column] / 3600, bins=40, color='C0', edgecolor='black')
    ax.grid(True)
    ax.set_yscale('log')
    ax.set_ylabel('Количество задач', fontsize=12)
    ax.set_xlabel(title.split(' (')[0] + ', ч', fontsize=12)
    if 'Среднее' in title:
        ax.set_title('З' + title.split(' (з')[1].strip('()'), fontsize=15)
    ax.set_ylim([1, None])


plt.tight_layout()
plt.show()
plt.savefig('pictures_for_diploma/mean_median_max_wait_time_for_wait_run.png')

X_train, X_test, y_train, y_test = train_test_split(dataset_f, y_vector, shuffle=False)
columns_to_scale = ['time_wait_mean', 'time_wait_median', 'time_wait_max',
       'now_time_run_mean', 'now_time_run_median', 'now_time_run_max',
       'now_time_wait_mean', 'now_time_wait_median', 'now_time_wait_max',
       'wait_count', 'run_count', 'wait_nodes_sum', 'run_tasks_sum',
       'run_user_count', 'run_group_count', 'query_user_count',
       'query_group_count', 'run_user_nodes', 'run_group_nodes',
       'query_user_req_tasks', 'query_group_req_tasks', 'wallclock_limit',
       'tasks_requested']

scaler = StandardScaler()
X_train.loc[:, columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test.loc[:, columns_to_scale] = scaler.transform(X_test[columns_to_scale])

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test = target_scaler.transform(y_test.values.reshape(-1, 1))

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

X_train.to_pickle('data/dataset_2/X_train.pickle')
X_test.to_pickle('data/dataset_2/X_test.pickle')
pd.Series(y_train.ravel()).to_pickle('data/dataset_2/y_train.pickle')
pd.Series(y_test.ravel()).to_pickle('data/dataset_2/y_test.pickle')


with open('data/dataset_2/target_scaler.pickle', 'wb') as f:
    pickle.dump(target_scaler, f)