import joblib
import pandas as pd
from prepare_data import prepare

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

## Dataset class
class MoADataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self): return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {'x': torch.tensor(self.features[idx, :], dtype=torch.float),
               'y': torch.tensor(self.targets [idx, :], dtype=torch.float)}
        return dct

class TestDataset(Dataset):
    def __init__(self, features):
        self.features = features
        
    def __len__(self): return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {'x': torch.tensor(self.features[idx, :], dtype=torch.float)}
        return dct


## Lightning data module
class MoADataModule(pl.LightningDataModule):
    def __init__(self, data_dir, fold, out_path=None, fe=False):
        super().__init__()
        self.data_dir = data_dir
        self.fold = fold
        self.drive_path = data_dir if out_path == None else out_path
        self.fe = fe

    def prepare_data(self):
        if not (self.drive_path/'folds.csv').exists():
            print('NO folds.csv found. . . . creating it!')
            train_features       = pd.read_csv(self.data_dir/'train_features.csv')
            train_targets_scored = pd.read_csv(self.data_dir/'train_targets_scored.csv')
            test_features        = pd.read_csv(self.data_dir/'test_features.csv')
            drug                 = pd.read_csv(self.data_dir/'train_drug.csv')

            targets = train_targets_scored.columns[1:]
            train_targets_scored = train_targets_scored.merge(drug, on='sig_id', how='left') 

            folds, test, feature_cols, target_cols = prepare(train_features, test_features, train_targets_scored, targets, self.fe)
            
            self.drive_path.mkdir(exist_ok=True)
            
            folds.to_csv(self.drive_path/'folds.csv', index=False)
            test .to_csv(self.drive_path/'test.csv' , index=False)

            columns = {'features': feature_cols, 'targets': target_cols}
            joblib.dump(columns, self.drive_path/'columns.pkl')
            
            print(f'Done! . . . . path: {self.drive_path}')
        else:
            print(f'Already exists! . . . . path: {self.drive_path}')

    def setup(self, stage=0):
        # Train
        train = pd.read_csv(self.drive_path/'folds.csv')
        cols  = joblib.load(self.drive_path/'columns.pkl')
        feature_cols, target_cols = cols['features'], cols['targets']

        self.trn_idx = train[train['kfold'] != self.fold].index
        self.val_idx = train[train['kfold'] == self.fold].index

        train_df = train[train['kfold'] != self.fold].reset_index(drop=True)
        valid_df = train[train['kfold'] == self.fold].reset_index(drop=True)

        x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
        x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values

        self.train_dataset = MoADataset(x_train, y_train)
        self.valid_dataset = MoADataset(x_valid, y_valid)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=128, 
                          num_workers=2, 
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, 
                          batch_size=128, 
                          num_workers=2, 
                          pin_memory=True)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    path = sys.argv[1]
    path = Path(path)
    dm = MoADataModule(path, 2)
    dm.prepare_data()
    dm.setup()
