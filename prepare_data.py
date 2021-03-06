import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

## Data Utils functions
def RankGauss(train, test, cols):
    transformer = QuantileTransformer(n_quantiles=100, output_distribution="normal")
    train[cols] = transformer.fit_transform(train[cols])
    test [cols] = transformer.transform(test[cols])
    return train, test

def get_pca(train, test, cols, n_comp, prefix):
    data = pd.concat([train[cols], test[cols]])
    data2 = (PCA(n_components=n_comp).fit_transform(data[cols]))
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

def fe_cluster(train, test, n_clusters_g=35, n_clusters_c=5):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans = KMeans(n_clusters = n_clusters).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test  = pd.get_dummies(test , columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

def fe_stats(train, test):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
    return train, test

def process(train, test, fe):
    GENES = [col for col in train.columns if col.startswith('g-')]
    CELLS = [col for col in train.columns if col.startswith('c-')]
    
    # normalize data using RankGauss
    train, test = RankGauss(train, test, GENES + CELLS)
    
    # get PCA components
    train, test = get_pca(train, test, GENES, 600, 'G') # GENES
    train, test = get_pca(train, test, CELLS, 50 , 'C') # CELLS
    
    # select features uing variance thresholding
    train, test = select_features(train, test)
    
    if fe:
        print('Engineering New Features!')
        # feature engineering
        train, test = fe_cluster(train, test)
        train, test = fe_stats  (train, test)
    return train, test

def process_score(scored, targets, folds=7):
    # LOCATE DRUGS
    vc = scored.drug_id.value_counts()
    vc1 = vc.loc[vc<=18].index.sort_values()
    vc2 = vc.loc[vc>18].index.sort_values()

    # STRATIFY DRUGS 18X OR LESS
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True)
    tmp = scored.groupby('drug_id')[targets].mean().loc[vc1]
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)
    
    # STRATIFY DRUGS MORE THAN 18X
    skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True)
    tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)
    
    # ASSIGN FOLDS
    scored['kfold'] = scored.drug_id.map(dct1)
    scored.loc[scored.kfold.isna(),'kfold'] =\
        scored.loc[scored.kfold.isna(),'sig_id'].map(dct2)
    scored.kfold = scored.kfold.astype('int8')
    return scored

def prepare(train, test, scored, targets, fe=False):
    train, test = process(train, test, fe)
    scored = process_score(scored, targets)
    
    # merge features with scores
    folds = train.merge(scored, on='sig_id')
    folds = folds[folds['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test  = test [test ['cp_type']!='ctl_vehicle'].reset_index(drop=True)

    # folds = folds.drop('cp_type', axis=1)
    # test  = test.drop ('cp_type', axis=1)

    # converting column names to str
    folds.columns = [str(c) for c in folds.columns.values.tolist()]

    # One-hot encoding 
    folds = pd.get_dummies(folds, columns=['cp_time', 'cp_dose'])
    test  = pd.get_dummies(test , columns=['cp_time', 'cp_dose'])

    ## Targets
    target_cols = scored.drop(['sig_id', 'kfold', 'drug_id'], axis=1).columns.values.tolist()

    # features columns
    to_drop = target_cols + ['sig_id', 'kfold', 'drug_id','cp_type']
    feature_cols = [c for c in folds.columns if c not in to_drop]
    
    return folds, test, feature_cols, target_cols

if __name__ == "__main__":
    import sys
    import joblib
    from pathlib import Path
    
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path

    print(f'Reading data from {in_path} . . . ')
    train_features = pd.read_csv(in_path/'train_features.csv')
    test_features  = pd.read_csv(in_path/'test_features.csv')
    train_targets_scored = pd.read_csv(in_path/'train_targets_scored.csv')
    drug = pd.read_csv(in_path/'train_drug.csv')
    
    targets = train_targets_scored.columns[1:]
    train_targets_scored = train_targets_scored.merge(drug, on='sig_id', how='left') 

    print('Preparing data . . . ')
    folds, test, feature_cols, target_cols = prepare(train_features, 
                                                     test_features, 
                                                     train_targets_scored, 
                                                     targets)
    
    print(f'FOLDS: {folds.shape}')
    print(f'TEST: {test.shape}')
    print(f'Targets : {len(target_cols)}')
    print(f'Features : {len(feature_cols)}')
    
    print(f'Saving data to {out_path} . . . ')
    out_path.mkdir(exists=True)
    folds.to_csv(out_path/'folds.csv', index=False)
    test .to_csv(out_path/'test.csv' , index=False)
    columns = {'features': feature_cols, 'targets': target_cols}
    joblib.dump(columns, out_path/'columns.pkl')
