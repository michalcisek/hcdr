# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:51:19 2018

@author: mcisek001
"""
import pyarrow.feather as feather
import os
import numpy as np
from hyperopt import hp, tpe
from hyperopt.fmin import fmin

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from lightgbm import LGBMClassifier

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

def objective(params):
    params = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'learning_rate': '{:.3f}'.format(params['learning_rate']),
        'max_depth': int(params['max_depth'])
    }
    
    clf = LGBMClassifier(
        n_estimators=1000,
        **params)
    
    score = cross_val_score(clf, x_train, y_train, scoring='roc_auc', 
                            cv=KFold(n_splits=5, shuffle=True, random_state=56283)).mean()
    print("AUC {:.3f} params {}".format(score, params))
    return -1*score

space = {
    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'learning_rate': hp.choice('learing_rate', np.arange(0.01, 0.21, 0.01)),
    'max_depth': hp.choice('max_depth', [4, 5, 6, 7, 8])    
}

#to make this work you need to have networkx library version 1.11
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=30)