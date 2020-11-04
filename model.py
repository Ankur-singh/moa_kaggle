import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MoANet(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.4)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.4)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x

class MoAModel(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self.net(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # Source: https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html#training-epoch-level-metrics
        self.log('train_loss', loss, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self.net(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)    
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}
        return [optimizer], [scheduler]