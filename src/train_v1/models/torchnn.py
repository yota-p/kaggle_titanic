import numpy as np
from omegaconf import DictConfig
from typing import Any  # List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import pytorch_lightning as pl


def get_optimizer(
        optimizer_name: str,
        optimizer_param: DictConfig,
        model: torch.nn) -> torch.optim.Optimizer:
    OPTIMIZER_DICT = {
        'Adam': torch.optim.Adam
    }
    if optimizer_param is None:
        optimizer_param = {}
    return OPTIMIZER_DICT[optimizer_name](model.parameters(), **optimizer_param)


def get_scheduler(
        scheduler_name: str,
        scheduler_param: DictConfig,
        optimizer: torch.optim.Optimizer) -> Any:
    SCHEDULER_DICT = {
        'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR
    }
    if scheduler_param is None:
        scheduler_param = {}
    return SCHEDULER_DICT[scheduler_name](optimizer, **scheduler_param)


def get_loss_function(
        loss_function_name: str,
        loss_function_param: DictConfig) -> torch.nn.modules.loss._Loss:
    LOSS_FUNCTION_DICT = {
        'SmoothBCEwLogits': SmoothBCEwLogits,
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss
    }
    if loss_function_param is None:
        loss_function_param = {}
    return LOSS_FUNCTION_DICT[loss_function_name](**loss_function_param)


class LitModelV1(pl.LightningModule):
    def __init__(self, all_feat_cols, target_cols, dropout_rate, hidden_size, optimizercfg, schedulercfg, lossfncfg):
        super().__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2)

        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, len(target_cols))

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

        self.optimizercfg = optimizercfg
        self.schedulercfg = schedulercfg
        self.lossfncfg = lossfncfg

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)

        x = torch.cat([x3, x4], 1)

        x = self.dense5(x)

        return x

    def configure_optimizers(self):
        optimizer = get_optimizer(
                        optimizer_name=self.optimizercfg.name,
                        optimizer_param=self.optimizercfg.param,
                        model=self)
        scheduler = get_scheduler(
                        scheduler_name=self.schedulercfg.name,
                        scheduler_param=self.schedulercfg.param,
                        optimizer=optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch['features'], batch['label']
        y_hat = self(x)
        loss_fn = get_loss_function(self.lossfncfg.name, self.lossfncfg.param)
        loss = loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['features'], batch['label']
        y_hat = self(x)
        loss_fn = get_loss_function(self.lossfncfg.name, self.lossfncfg.param)
        val_loss = loss_fn(y_hat, y)
        return val_loss

    def validation_epoch_end(self, outputs):
        # calculate average loss in this epoch
        # outputs is a list of whatever you returned in `validation_step`
        loss = torch.stack(outputs).mean()
        self.log("val_loss", loss)


class EarlyStopping:
    def __init__(self, patience=7, mode='max'):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        if self.mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == 'min':
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class ModelV1(nn.Module):
    def __init__(self, all_feat_cols, target_cols, dropout_rate, hidden_size):
        super(ModelV1, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2)

        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, len(target_cols))

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)

        x = torch.cat([x3, x4], 1)

        x = self.dense5(x)

        return x
