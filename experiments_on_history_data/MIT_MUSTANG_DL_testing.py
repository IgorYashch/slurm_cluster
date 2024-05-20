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

from sklearn.metrics import mean_squared_error as sklearn_mse
from sklearn.metrics import mean_absolute_percentage_error as sklearn_mape
from sklearn.metrics import mean_absolute_error as sklearn_mae

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import Trainer, seed_everything, LightningDataModule


from slurm_nn_data_preparation import SlurmDataModule, make_sequence
from slurm_nn_arch import Net, SlurmModule


datamodule = SlurmDataModule(batch_size=32, num_workers=1)

experiment_name = 'lstm_4'
base_path = f'nn_experiments/{experiment_name}/'

input_size_seq = 42
input_size_fcnn = 41

params = {
    'optimizer': {'lr' : 0.001},
    'net': {
        'type': 'lstm',
        'dim': 64,
        'input_size': input_size_seq,
        'fcnn': {
            'input_size': input_size_fcnn,
            'hidden_size': 16,
            'output_size': 8,
            'num_layers': 2, 
            'dropout': 0.05
        },
        'lstm': {
            'input_size': input_size_seq,
            'hidden_size': 32,
            'num_layers': 1,
            'bidirectional': True,
            'dropout': 0.0
        },
        'squeezer': { 'squeezer_type': 'pooling', 'pooling_types': ['mean'] },
        'classifier': { 'num_layers': 1, 'hidden_size': 8, 'dropout': 0.05 }
    }
}


# In[5]:


module = SlurmModule(params)
checkpoint_mape_callback = ModelCheckpoint(
    monitor='val_mape', mode='min', filename='mape-{epoch}-{step}-{val_mape:.4f}', save_top_k=-1,
    dirpath=os.path.join(base_path, 'logs', experiment_name, params['net']['type'])
)
checkpoint_loss_callback = ModelCheckpoint(
    monitor='val_loss', mode='min', filename='loss-{epoch}-{step}-{val_loss:.4f}', save_top_k=-1,
    dirpath=os.path.join(base_path, 'logs', experiment_name, params['net']['type'])
)
early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0, patience=3, verbose=False)

trainer = Trainer(
    min_epochs=2,
    max_epochs=30,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=1,
    
    accelerator='cpu',
    deterministic=True,
    callbacks=[
        checkpoint_mape_callback,
        checkpoint_loss_callback,
        early_stopping_callback
    ]
)


test_preds = trainer.predict(
    module,
    dataloaders=datamodule.test_dataloader(), return_predictions=True, 
    ckpt_path='nn_experiments/lstm_5/logs/lstm_5/lstm/loss-epoch=12-step=59761-val_loss=0.2662.ckpt'
)

train_preds = trainer.predict(
    module,
    dataloaders=datamodule.train_dataloader(), return_predictions=True, 
    ckpt_path='nn_experiments/lstm_5/logs/lstm_5/lstm/loss-epoch=12-step=59761-val_loss=0.2662.ckpt'
)

with open('data/version_2/target_scaler.pickle', 'rb') as f:
    target_scaler = pickle.load(f)

train_preds_lstm = torch.cat([t[0] for t in train_preds])
train_targets_lstm = torch.cat([t[1] for t in train_preds])

preds_lstm = torch.cat([t[0] for t in test_preds])
targets_lstm = torch.cat([t[1] for t in test_preds])

def figure_preds(y_test, y_pred):
    plt.figure(figsize=(10, 3))
    plt.plot(y_test, label='wait time')
    plt.plot(y_pred, label='predicted time')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, len(y_test)])

def figure_preds_real(y_test, y_pred):
    y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(len(y_test)), y_test, label='wait time')
    plt.plot(np.arange(len(y_test)), y_pred, label='predicted time')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, len(y_test)])

figure_preds_real(
    targets_lstm, preds_lstm
)


figure_preds_real(
    train_targets_lstm, train_preds_lstm
)

mse = sklearn_mse(targets_lstm, preds_lstm)
mape = sklearn_mape(targets_lstm, preds_lstm)
mae = sklearn_mae(targets_lstm, preds_lstm)

train_mse = sklearn_mse(train_targets_lstm, train_preds_lstm)
train_mape = sklearn_mape(train_targets_lstm, train_preds_lstm)
train_mae = sklearn_mae(train_targets_lstm, train_preds_lstm)