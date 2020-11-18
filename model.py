import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.nn.modules.loss import _WeightedLoss

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

def BnDropLin(hs1, hs2):
    return [nn.BatchNorm1d(hs1),
            nn.Dropout(0.2),
            nn.utils.weight_norm(nn.Linear(hs1, hs2)),
            nn.ReLU()]

class MoANet(nn.Module):
    def __init__(self, num_features, num_targets, hidden_sizes):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_sizes[0]))
        
        layers = []
        for hs1, hs2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers += BnDropLin(hs1, hs2)    
        self.layers = nn.Sequential(*layers)
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_sizes[-1])
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_sizes[-1], num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x), 1e-3)
        
        x = self.layers(x)
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x

class MoAModel(pl.LightningModule):
    def __init__(self, net, config):
        super().__init__()
        self.net = net
        self.config = config
        self.loss_tr = SmoothBCEwLogits(smoothing =0.001) 
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self.net(x)
        loss = self.loss_tr(y_hat, y)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # Source: https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html#training-epoch-level-metrics
        self.log('train_loss', loss, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self.net(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)    
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        # scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                            pct_start=0.1,
                                                            div_factor=1e3, 
                                                            max_lr=1e-2, 
                                                            steps_per_epoch=self.config.steps_per_epochs, 
                                                            epochs=self.config.epochs)
                                                            
        scheduler = {'scheduler': lr_scheduler, 'interval': 'step', 'monitor': 'val_loss'}
        return [optimizer], [scheduler]
