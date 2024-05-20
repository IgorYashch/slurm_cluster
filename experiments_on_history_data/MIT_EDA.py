import pandas as pd
import numpy as np
import seaborn as sb
import tqdm
import glob
import time
import pickle
import ast
import os

from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

scheduler_data = pd.read_csv('./data/scheduler_data.csv')
scheduler_data.columns

unique_users = np.unique(scheduler_data['id_user'])

np.unique(scheduler_data['array_task_pending'])


set(scheduler_data['partition']) 

pd.isna(scheduler_data['gres_req'][0])


scheduler_data['gres_req_num'] = scheduler_data['gres_req'].apply(lambda s: 0 if pd.isna(s) else s.split(':')[-1]).astype(int)
scheduler_data['gres_req_type'] = scheduler_data['gres_req'].apply(
    lambda s: -1 if pd.isna(s) else int('volta' in s)
).astype(int)



scheduler_data['gres_alloc_num'] = scheduler_data['gres_alloc'].apply(lambda s: 0 if pd.isna(s) else s.split(':')[-1]).astype(int)
scheduler_data['gres_alloc_type'] = scheduler_data['gres_alloc'].apply(
    lambda s: -1 if pd.isna(s) else int('volta' in s)
).astype(int)


scheduler_data.drop(columns='time_suspended', inplace=True)
scheduler_data['time_wait'] = scheduler_data['time_start'] - scheduler_data['time_submit']
scheduler_data = scheduler_data[scheduler_data.time_start > 0].reset_index(drop=True)

scheduler_data_1, scheduler_data_2 = train_test_split(scheduler_data, shuffle=False)

scheduler_data.job_type.value_counts()

scheduler_data.groupby('job_type').time_wait.mean()


scheduler_data.groupby('job_type').time_wait.median()

wait_time_train, wait_time_test = train_test_split(
    scheduler_data.reset_index()['time_wait'],
    shuffle=False
)
train_index = wait_time_train.index
test_index = wait_time_test.index


plt.figure(figsize=(12, 4))
plt.plot(train_index, wait_time_train, label='Обучающая выборка')
plt.plot(test_index, wait_time_test, label='Тестовая выборка')
plt.legend()
plt.grid(True)
plt.ylim([0, None])
plt.xlim([0, len(scheduler_data)])
plt.ylabel('Время ожидания, с', fontsize=12)
plt.xlabel('Индекс задачи', fontsize=12)
plt.savefig('pictures_for_diploma/wait_time_plot.png')

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


def make_new_features(scheduler_data, row):

    new_features_dict = {}
    cur_time_submit = row['time_submit']

    mask_1 = scheduler_data.time_submit < cur_time_submit
    mask_2 = scheduler_data.time_start > cur_time_submit
    mask_3 = scheduler_data.time_start < cur_time_submit
    mask_4 = scheduler_data.time_end > cur_time_submit

    sched_1 = scheduler_data[mask_1 & mask_2].copy()
    sched_2 = scheduler_data[mask_3 & mask_4].copy()

    sched_1['now_time_wait'] = cur_time_submit - sched_1.time_submit
    sched_2['now_time_run'] = cur_time_submit - sched_2.time_start
    
    for col in [sched_2['time_wait'], sched_2['now_time_run'], sched_1['now_time_wait']]:
        new_features_dict[col.name + '_mean'] = col.mean()
        new_features_dict[col.name + '_median'] = col.median()
        new_features_dict[col.name + '_max'] = col.max()
        
    new_features_dict['run_count'] = len(sched_1)
    new_features_dict['wait_count'] = len(sched_2)

    last_submited = scheduler_data[mask_1].reset_index(drop=True)[-10:]
    last_submited['now_time_wait'] = cur_time_submit - last_submited.time_submit
    last_submited['real_wait_time'] = [min(a, b) for a, b in zip(
        last_submited['now_time_wait'], last_submited['time_wait']
    )]
    
    
    new_features_dict['run_user_count'] = len(sched_2[sched_2["id_user"] == row["id_user"]])
    new_features_dict['wait_user_count'] = len(sched_1[sched_1["id_user"] == row["id_user"]])
    
#     nodes_alloc / gres_alloc

    new_features_dict['run_user_nodes_alloc'] = sched_2[sched_2["id_user"] == row["id_user"]].nodes_alloc.sum()
    new_features_dict['run_user_gres_alloc'] = sched_2[sched_2["id_user"] == row["id_user"]].gres_alloc_num.sum()
    
#     cpus_req / gres_req / mem_req

    new_features_dict['wait_user_cpus_req'] = sched_1[sched_1["id_user"] == row["id_user"]].cpus_req.sum()
    new_features_dict['wait_user_gres_req'] = sched_1[sched_1["id_user"] == row["id_user"]].gres_req_num.sum()
    new_features_dict['wait_user_mem_req'] = sched_1[sched_1["id_user"] == row["id_user"]].mem_req.sum()

    last_5_wait_time = last_submited[-5:].real_wait_time.median()
    last_10_wait_time = last_submited[-10:].real_wait_time.median()


    new_features_dict['last_5_wait_time'] = last_submited[-5:].real_wait_time.median()
    new_features_dict['last_10_wait_time'] = last_submited[-10:].real_wait_time.median()
    
    return new_features_dict

len(scheduler_data.time_submit)



scheduler_data['gres_req'] = scheduler_data['gres_req'].fillna(0)
dataset_f = {}
for i, row in tqdm.tqdm(enumerate(scheduler_data.iterrows())):
    dataset_f[i] = make_new_features(scheduler_data, row[1])

dataset_f['timelimit'] = scheduler_data['timelimit'].values
dataset_f['mem_req'] = scheduler_data['mem_req'].values
dataset_f['cpus_req'] = scheduler_data['cpus_req'].values

dataset_f['partition'] = scheduler_data['partition'].values
dataset_f['job_type'] = scheduler_data['job_type'].values

dataset_f = pd.DataFrame(dataset_f).T.fillna(0)

scheduler_data.timelimit.isna().sum(), scheduler_data.mem_req.isna().sum(), scheduler_data.cpus_req.isna().sum()

y_vector = scheduler_time_data.time_wait

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].hist(dataset_f.last_5_wait_time, bins=40, color='C0', edgecolor='black')
axs[0].grid(True)
axs[0].set_xlabel('Медианное время ожидания для последних 5 задач, с', fontsize=12)
axs[0].set_ylabel('Количество задач', fontsize=12)
axs[0].set_yscale('log')

axs[1].hist(dataset_f.last_10_wait_time, bins=40, color='C0', edgecolor='black')
axs[1].grid(True)
axs[1].set_xlabel('Медианное время ожидания для последних 10 задач, с', fontsize=12)
axs[1].set_ylabel('Количество задач', fontsize=12)
axs[1].set_yscale('log')

plt.tight_layout()
plt.show()

plt.savefig('pictures_for_diploma/median_wait_time_5_10_last.png')

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

bins = set([np.quantile(dataset_f.timelimit, q) for q in np.arange(0.1, 1, 0.01)])
bins = sorted(bins)

bins_mem_req = set([np.quantile(dataset_f.mem_req, q) for q in np.arange(0.1, 1, 0.01)])
bins_mem_req = sorted(bins_mem_req)

dataset_f.timelimit.isna().sum()

dataset_f['mem_req_cat'] = np.digitize(dataset_f.mem_req, bins_mem_req)
dataset_f['timelimit_cat'] = np.digitize(dataset_f.timelimit, bins)

features_cats = dataset_f.copy()
cats_f = ['timelimit_cat', 'mem_req_cat', 'job_type', 'gres_req_type', 'partition']
features_cats[cats_f] = features_cats[cats_f].astype('category')

dataset_f = pd.get_dummies(
    dataset_f,
    columns=cats_f,
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(dataset_f, y_vector, shuffle=False)
X_train_cats, X_test_cats, y_train, y_test = train_test_split(features_cats, y_vector, shuffle=False)

columns_to_scale = ['timelimit', 'time_wait_mean', 'time_wait_median', 'time_wait_max',
       'now_time_run_mean', 'now_time_run_median', 'now_time_run_max',
       'now_time_wait_mean', 'now_time_wait_median', 'now_time_wait_max',
       'run_count', 'wait_count',
       'last_5_wait_time', 'last_10_wait_time',  'mem_req', 'cpus_req'] #

scaler = StandardScaler()
X_train.loc[:, columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test.loc[:, columns_to_scale] = scaler.transform(X_test[columns_to_scale])

X_train_cats.loc[:, columns_to_scale] = scaler.fit_transform(X_train_cats[columns_to_scale])
X_test_cats.loc[:, columns_to_scale] = scaler.transform(X_test_cats[columns_to_scale])

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test = target_scaler.transform(y_test.values.reshape(-1, 1))

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

X_train_cats = X_train_cats.reset_index(drop=True)
X_test_cats = X_test_cats.reset_index(drop=True)

X_train_cats.to_pickle('data/version_2/X_train_cats.pickle')
X_test_cats.to_pickle('data/version_2/X_test_cats.pickle')
X_train.to_pickle('data/version_2/X_train.pickle')
X_test.to_pickle('data/version_2/X_test.pickle')
pd.Series(y_train.ravel()).to_pickle('data/version_2/y_train.pickle')
pd.Series(y_test.ravel()).to_pickle('data/version_2/y_test.pickle')

with open('data/version_2/target_scaler.pickle', 'wb') as f:
    pickle.dump(target_scaler, f)
