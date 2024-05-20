import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from lightning.pytorch import Trainer, seed_everything, LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


def make_sequence(cur_time_submit, time_data, data_features):
    mask_1 = time_data.time_submit < cur_time_submit
    mask_2 = time_data.time_start > cur_time_submit
    mask_3 = time_data.time_start < cur_time_submit
    mask_4 = time_data.time_end > cur_time_submit
    
    waiting_tasks = data_features[mask_1 & mask_2].copy()
    runing_tasks = data_features[mask_3 & mask_4].copy()
    
    waiting_tasks['is_run'] = [0] * len(waiting_tasks)
    runing_tasks['is_run'] = [1] * len(runing_tasks)
    
    sequence = pd.concat([waiting_tasks, runing_tasks])
    if len(sequence) == 0:
        sequence.loc[0] = [0] * sequence.shape[1]
    
    return sequence.values


def collate_function(batch):

    seq_items, fcnn_items, targets, mask = [], [], [], []
    for (seq_item, fcnn_item), target in batch:
        seq_items.append(seq_item)
        fcnn_items.append(fcnn_item)
        targets.append(target)
        mask.append(torch.full(size=(len(seq_item),), fill_value=True))

    seq_items = torch.nn.utils.rnn.pad_sequence(seq_items, batch_first=True, padding_value=0).type(torch.FloatTensor)
    fcnn_items = torch.stack(fcnn_items).type(torch.FloatTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=False).type(torch.FloatTensor)
    targets = torch.Tensor(targets).type(torch.FloatTensor)

    return dict(seq_tensor=seq_items, fcnn_tensor=fcnn_items, mask=mask, targets=targets)


class SlurmDataset(Dataset):

    def __init__(self, data_features, time_data, targets, idx_col):
        self.time_data = time_data
        self.data_features = data_features
        self.targets = targets
        self.idx_col = idx_col

    def __getitem__(self, index):
        i = self.idx_col[index]
        seq_item = torch.from_numpy(make_sequence(i, self.time_data, self.data_features))
        fcnn_item = torch.from_numpy(self.data_features.values[i])
        return (seq_item, fcnn_item), torch.FloatTensor([self.targets[i]])

    def __len__(self):
        return len(self.idx_col)
    
class SlurmDatasetLoad(Dataset):

    def __init__(self, data_features, targets, idx_col):
        self.data_features = data_features
        self.targets = targets
        self.idx_col = idx_col

    def __getitem__(self, index):
        i = self.idx_col[index]
        seq_item = torch.load(f'torch_objs_train/obj_{i}.pickle')
        fcnn_item = torch.from_numpy(self.data_features.values[i])
        return (seq_item, fcnn_item), torch.FloatTensor([self.targets[i]])

    def __len__(self):
        return len(self.idx_col)


class SlurmDataModule(LightningDataModule):
    
    def __init__(self, batch_size=32, num_workers=0):

        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers

        scheduler_data = pd.read_csv('data/scheduler_data.csv')
        scheduler_data = scheduler_data[scheduler_data.time_start > 0].reset_index(drop=True)
        
        self.time_data = scheduler_data[['time_submit', 'time_eligible', 'time_start', 'time_end']].copy()
        
        self.data_features = pd.concat(
            [pd.read_pickle('data/version_2/X_train.pickle'), pd.read_pickle('data/version_2/X_test.pickle')]
        ).reset_index(drop=True)
        
        self.train_val_size = int(len(self.data_features) * 0.75)
        self.test_size = len(self.data_features) - self.train_val_size
        
        self.idx_col = self.time_data['time_submit'].values
        
        self.targets = np.concatenate(
            [pd.read_pickle('data/version_2/y_train.pickle').values,
             pd.read_pickle('data/version_2/y_test.pickle').values]
        )
        
        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None
        
        self.test_idx = np.arange(self.test_size) + self.train_val_size
        self.train_idx, self.valid_idx = train_test_split(
            np.arange(self.train_val_size), shuffle=True, random_state=42, test_size=0.1
        )

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        
        self.train_dataset = SlurmDatasetLoad(
            self.data_features, self.targets, self.train_idx
        )
        
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, collate_fn=collate_function, persistent_workers=True
        )

    def val_dataloader(self):
        self.valid_dataset = SlurmDatasetLoad(
            self.data_features, self.targets, self.valid_idx
        )
        return DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_function, persistent_workers=True
        )

    def test_dataloader(self):
        self.test_dataset = SlurmDatasetLoad(
            self.data_features, self.targets, self.test_idx
        )
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_function, persistent_workers=True
        )

    def teardown(self, stage: str):
        if stage == 'fit':
            self.train_dataset = None
            self.valid_dataset = None
        if stage == 'test':
            self.test_dataset = None
            