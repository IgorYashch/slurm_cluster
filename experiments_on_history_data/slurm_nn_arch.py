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

from sklearn.metrics import mean_squared_error as sklearn_mse
from sklearn.metrics import mean_absolute_percentage_error as sklearn_mape
from sklearn.metrics import mean_absolute_error as sklearn_mae


class PreSequentialLayer(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
    def forward(self, batch):
        return dict(tensor=batch['seq_tensor'], mask=batch['mask'], targets=batch['targets'])

class FullyConnectedLayer(pl.LightningModule):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        
        super().__init__()
        
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(p=dropout)]
        cur_size = hidden_size
        for i in range(num_layers - 2):
            layers.append(nn.Linear(cur_size, cur_size // 2))
            cur_size = cur_size // 2
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(cur_size, output_size))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, batch):
        next_state = self.layers(batch['fcnn_tensor'])
        return next_state

class PreDataBlock(pl.LightningModule):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, batch):
        next_state = self.linear(batch['tensor'])
        return dict(tensor=next_state, mask=batch['mask'], targets=batch['targets'])


class SelfAttentionLayer(pl.LightningModule):

    def __init__(self, input_size, num_heads=8, dropout=0.0):
        
        super().__init__()
        self.cls_token = nn.Parameter(torch.rand(input_size))
        self.mha = nn.MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, batch):
        batch_size = batch['tensor'].shape[0]
        with_token = torch.cat([self.cls_token.repeat((batch_size, 1, 1)), batch['tensor']], dim=1)
        new_mask = torch.cat([torch.full((batch_size, 1), fill_value=True), batch['mask']], dim=1).to(torch.bool)
        next_state, _ = self.mha(with_token, with_token, with_token, key_padding_mask=~new_mask)
        return dict(tensor=next_state, mask=new_mask, targets=batch['targets'])


class LSTM(pl.LightningModule):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0):
        
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True,
            num_layers=num_layers, bidirectional=bidirectional, dropout=dropout
        )
        
    def forward(self, batch):
        lengths = batch['mask'].sum(-1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            batch['tensor'], lengths, batch_first=True, enforce_sorted=False
        )
        output, (h, c) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return dict(tensor=output, mask=batch['mask'], targets=batch['targets'])

class GRU(pl.LightningModule):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0):
        
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, batch_first=True,
            num_layers=num_layers, bidirectional=bidirectional, dropout=dropout
        )
        
    def forward(self, batch):
        lengths = batch['mask'].sum(-1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            batch['tensor'], lengths, batch_first=True, enforce_sorted=False
        )
        output, h = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return dict(tensor=output, mask=batch['mask'], targets=batch['targets'])

class ResidualLSTM(pl.LightningModule):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0):
        
        super().__init__()
        self.lstm = LSTM(
            input_size, hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout
        )
        D = 2 if bidirectional else 1
        self.linear =  nn.Sequential(
            nn.Linear(hidden_size * D, input_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
    def forward(self, batch):
        
        x = batch['tensor']
        lstm_out = self.lstm(batch)['tensor']
        after_linear = self.linear(lstm_out)
        next_state = x + after_linear
        return dict(tensor=next_state, mask=batch['mask'], targets=batch['targets'])
    
    
class Squeezer(pl.LightningModule):

    def __init__(self, squeezer_type='positional', pos=0, pooling_types=['mean']):
        
        super().__init__()

        assert squeezer_type in ['positional', 'pooling']
        assert isinstance(pos, int)
        assert isinstance(pooling_types, list)

        self.squeezer_type = squeezer_type
        self.pos = pos
        self.pooling_types = pooling_types
        
    def forward(self, batch):
        if self.squeezer_type == 'positional':
            next_state = batch['tensor'][:, self.pos, :]
        elif self.squeezer_type == 'pooling':
            poolings = []
            for pooling_type in self.pooling_types:
                if pooling_type == 'mean':
                    poolings.append(batch['tensor'].mean(1))
                elif pooling_type == 'max':
                    poolings.append(batch['tensor'].max(1)[0])
                elif pooling_type == 'min':
                    poolings.append(batch['tensor'].min(1)[0])
            next_state = torch.cat(poolings, dim=-1)
        return next_state
#         return dict(tensor=next_state, targets=batch['targets'])


class RegressionLayer(pl.LightningModule):

    def __init__(self, input_size, num_layers, hidden_size, dropout=0.0):
        
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(p=dropout)]
        cur_size = hidden_size
        for i in range(num_layers - 2):
            layers.append(nn.Linear(cur_size, cur_size // 2))
            cur_size = cur_size // 2
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(cur_size, 1))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, batch):
        next_state = self.layers(batch['tensor']).flatten()
        return dict(tensor=next_state, targets=batch['targets'])


class Net(pl.LightningModule):

    def __init__(self, params):

        super().__init__()
        self.params = params
        self.configure_net(params)

    def forward(self, batch):
        seq_emb = self.seq_layers(batch)
        fcnn_emb = self.fcnn_layers(batch)
        concated = torch.cat([seq_emb, fcnn_emb], dim=-1)
        next_state = dict(tensor=concated, targets=batch['targets'])
        return self.final_layer(next_state)
    
    def configure_net(self, params):
        
        embed_dim = params['dim']
        input_size = params['input_size']
        fcnn_params = params['fcnn']
        
        seq_layers = [PreSequentialLayer()]

        if params['type'] == 'lstm':
            lstm_param = params['lstm']
            seq_layers.append(LSTM(**lstm_param))
            embed_dim = lstm_param['hidden_size'] * (2 if lstm_param['bidirectional'] else 1)
        elif params['type'] == 'attention':
            seq_layers.append(PreDataBlock(input_size=input_size, output_size=params['attention']['input_size']))
            seq_layers.append(SelfAttentionLayer(**params['attention']))
        elif params['type'] == 'residual_lstm':
            seq_layers.append(ResidualLSTM(**params['lstm']))
        elif params['type'] == 'gru':
            lstm_param = params['lstm']
            seq_layers.append(GRU(**lstm_param))
            embed_dim = lstm_param['hidden_size'] * (2 if lstm_param['bidirectional'] else 1)
        
        if params['squeezer']['squeezer_type'] == 'positional':
            after_squeezer_dim = embed_dim
        elif params['squeezer']['squeezer_type'] == 'pooling':
            after_squeezer_dim = embed_dim * len(params['squeezer']['pooling_types'])
            
        seq_layers.append(Squeezer(**params['squeezer']))
        
        self.seq_layers = nn.Sequential(*seq_layers)
        self.fcnn_layers = FullyConnectedLayer(**fcnn_params)
        self.final_layer =  RegressionLayer(
            input_size=after_squeezer_dim + fcnn_params['output_size'],
            **params['classifier']
        )
    
    
class SlurmModule(pl.LightningModule):
  
    def __init__(self, params):
        
        super().__init__()
        self.params = params
        self.net = Net(params['net'])
        self.loss_func = nn.MSELoss()
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch):
        return self.net(batch)

    def training_step(self, batch, batch_idx):
        logits = self(batch)['tensor']
        loss = self.loss_func(logits, batch['targets'])
        self.log('train_loss', loss, prog_bar=True)
        self.training_step_outputs.append(torch.stack([logits, batch['targets']], dim=-1))
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)['tensor']
        loss = self.loss_func(logits, batch['targets'])
        self.log('val_loss', loss, prog_bar=True)
        self.validation_step_outputs.append(torch.stack([logits, batch['targets']], dim=-1))
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch)['tensor']
        loss = self.loss_func(logits, batch['targets'])
        self.log('test_loss', loss, prog_bar=True)
        self.test_step_outputs.append(torch.stack([logits, batch['targets']], dim=-1))
        return loss
    
    def predict_step(self, batch, batch_idx):
        logits = self(batch)['tensor']
        return logits, batch['targets']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params['optimizer']['lr'])
        return optimizer
        
    def on_train_epoch_end(self):
        all_preds = torch.cat(self.training_step_outputs, dim=0).detach().cpu()
        mse = sklearn_mse(all_preds[:, 1], all_preds[:, 0])
        mape = sklearn_mape(all_preds[:, 1], all_preds[:, 0])
        self.log('train_mape', mape, prog_bar=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs, dim=0).detach().cpu()
        mse = sklearn_mse(all_preds[:, 1], all_preds[:, 0])
        mape = sklearn_mape(all_preds[:, 1], all_preds[:, 0])
        self.log('val_mape', mape, prog_bar=True)
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_step_outputs, dim=0).detach().cpu()
        mse = sklearn_mse(all_preds[:, 1], all_preds[:, 0])
        mape = sklearn_mape(all_preds[:, 1], all_preds[:, 0])
        self.log('test_mape', mape, prog_bar=True)
        self.test_step_outputs.clear()