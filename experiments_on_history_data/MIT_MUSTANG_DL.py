import pandas as pd
import numpy as np
import tqdm
import glob
import time
import pickle
import os
import torchviz
import re

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


# from slurm_nn_data_preparation import SlurmDataModule
from data_prep_old import SlurmDataModule
from slurm_nn_arch import Net, SlurmModule

experiment_name = 'lstm_5'
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


trainer.fit(
    model=module, datamodule=datamodule,
    ckpt_path='nn_experiments/lstm_5/logs/lstm_5/lstm/loss-epoch=8-step=41373-val_loss=0.3125.ckpt'
)

loss_paths = glob.glob('nn_experiments/lstm_5/logs/lstm_5/lstm/loss*')
mape_paths = glob.glob('nn_experiments/lstm_5/logs/lstm_5/lstm/mape*')

losses = [float(re.search(r'\d\.\d+', s)[0]) for s in loss_paths]
mapes = [float(re.search(r'\d\.\d+', s)[0]) for s in mape_paths]


plt.figure(figsize=(15, 5))
plt.plot(losses, '--o', label='MSE Loss')
plt.grid(True)
plt.legend()
plt.ylim([0, None])

plt.figure(figsize=(15, 5))
plt.plot(mapes, '--o', label='MAPE')
plt.grid(True)
plt.ylim([0, None])
