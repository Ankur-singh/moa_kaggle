import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

## Data Utils functions
def RankGauss(train, test, cols):
    transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
    train[cols] = transformer.fit_transform(train[cols])
    test [cols] = transformer.transform(test[cols])
    return train, test

def get_pca(train, test, cols, n_comp, prefix):
    data = pd.concat([train[cols], test[cols]])
    data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[cols]))
    train2 = data2[:train.shape[0]] 
    test2  = data2[-test.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_{prefix}-{i}' for i in range(n_comp)])
    test2  = pd.DataFrame(test2 , columns=[f'pca_{prefix}-{i}' for i in range(n_comp)])

    train = pd.concat((train, train2), axis=1)
    test  = pd.concat((test , test2 ), axis=1)
    return train, test

def select_features(train, test):
    var_thresh = VarianceThreshold(0.8)  #<-- Update
    data = train.append(test)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_transformed = data_transformed[ : train.shape[0]]
    test_transformed  = data_transformed[-test.shape[0] : ]

    train = pd.DataFrame(train[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                  columns=['sig_id','cp_type','cp_time','cp_dose'])

    train = pd.concat([train, pd.DataFrame(train_transformed)], axis=1)


    test = pd.DataFrame(test[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                 columns=['sig_id','cp_type','cp_time','cp_dose'])

    test = pd.concat([test, pd.DataFrame(test_transformed)], axis=1)
    return train, test

def process(train, test):
    GENES = [col for col in train.columns if col.startswith('g-')]
    CELLS = [col for col in train.columns if col.startswith('c-')]
    
    # normalize data using RankGauss
    train, test = RankGauss(train, test, GENES + CELLS)
    
    # get PCA components
    train, test = get_pca(train, test, GENES, 600, 'G') # GENES
    train, test = get_pca(train, test, CELLS, 50 , 'C') # CELLS
    
    # select features uing variance thresholding
    train, test = select_features(train, test)
    return train, test

def prepare(train, test, train_scored):
    train, test = process(train, test)
    
    # merge features with scores
    train = train.merge(train_scored, on='sig_id')
    train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test  = test [test ['cp_type']!='ctl_vehicle'].reset_index(drop=True)

    ## Targets
    target = train[train_scored.columns]
    target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

    ## features
    train = train.drop('cp_type', axis=1)
    test  = test.drop ('cp_type', axis=1)
    
    # create folds
    folds = train.copy()
    folds.columns = [str(c) for c in folds.columns.values.tolist()]

    mskf = MultilabelStratifiedKFold(n_splits=7)

    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        folds.loc[v_idx, 'kfold'] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)
    
    # One-hot encoding 
    folds = pd.get_dummies(folds, columns=['cp_time', 'cp_dose'])
    test  = pd.get_dummies(test , columns=['cp_time', 'cp_dose'])

    # features columns
    to_drop = target_cols + ['kfold','sig_id']
    feature_cols = [c for c in folds.columns if c not in to_drop]
    
    return folds, test, feature_cols, target_cols

if __name__ == "__main__":
    import sys
    import joblib
    
    path = sys.argv[1]

    train_features = pd.read_csv(f'{path}/train_features.csv')
    test_features  = pd.read_csv(f'{path}/test_features.csv')
    train_targets_scored = pd.read_csv(f'{path}/train_targets_scored.csv')

    folds, test, feature_cols, target_cols = prepare(train_features, test_features, train_targets_scored)
    
    print(folds.shape)
    print(test.shape)
    print(f'Targets : {len(target_cols)}')
    print(f'Features : {len(feature_cols)}')
    
    folds.to_csv(path/'folds.csv', index=False)
    test .to_csv(path/'test.csv' , index=False)
    columns = {'features': feature_cols, 'targets': target_cols}
    joblib.dump(columns, path/'columns.pkl')
