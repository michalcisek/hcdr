# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:43:24 2018

@author: mcisek001
"""
import numpy as np
import pandas as pd
import os
import pyarrow.feather as feather

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

full_dataset = True

if full_dataset:
    filePathTRAIN = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TRAIN_sample.feather')
    filePathTEST = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TEST_sample.feather')
else:
    filePathTRAIN = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TRAIN_sample_afterFS.feather')
    filePathTEST = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TEST_sample_afterFS.feather')
    

train = feather.read_feather(filePathTRAIN)
test = feather.read_feather(filePathTEST)

train = train.fillna(0)

y_train = train["TARGET"]

#y_train = y_train.values
x_train = train.drop(["TARGET"], axis=1)

#check if columns contain Inf values and drop them
inf_sum = np.isinf(x_train).sum()
inf_sum = inf_sum.index[np.where(inf_sum > 0)]

x_train.drop(columns = list(inf_sum), inplace = True)
test.drop(columns = list(inf_sum), inplace = True)

colnames = x_train.columns.values

oof_preds = np.zeros(x_train.shape[0])
sub_preds = np.zeros(test.shape[0])

feature_importance_df = pd.DataFrame()

feats = [f for f in x_train.columns if f not in ['SK_ID_CURR']]

folds = KFold(n_splits=5, shuffle=True, random_state=56283)
    
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train)):
    trn_x, trn_y = x_train[feats].iloc[trn_idx], y_train.iloc[trn_idx]
    val_x, val_y = x_train[feats].iloc[val_idx], y_train.iloc[val_idx]
    
    #clf = LGBMClassifier(n_estimators=4000, learning_rate=0.03,
    #    num_leaves=30, colsample_bytree=.8,
    #    subsample=.9, max_depth=7, reg_alpha=.1,
    #    reg_lambda=.1, min_split_gain=.01,
    #    min_child_weight=2, silent=-1, verbose=-1)
    
    clf = LGBMClassifier(n_estimators=4000, learning_rate=0.03,
        num_leaves=112, colsample_bytree=.38,
        subsample=.9, max_depth=6, silent=-1, verbose=-1)
    
    clf.fit(trn_x, trn_y, eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=100)
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    
print('Full AUC score %.6f' % roc_auc_score(y_train, oof_preds)) 

test['TARGET'] = sub_preds

submission = test[['SK_ID_CURR', 'TARGET']]

if full_dataset:
    submission.to_csv('lightgbm.csv', index=False)
else:
    submission.to_csv('lightgbm_fs.csv', index=False)
    
