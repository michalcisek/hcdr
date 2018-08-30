# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:22:55 2018

@author: mcisek001
"""

import numpy as np
import pandas as pd
import os
import pyarrow.feather as feather

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
import xgboost as xgb

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
test = test.drop(["TARGET"], axis=1)
 

clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1,
    gamma=0.17, colsample_bytree=0.45, max_depth=7, silent=-1, verbose=-1)

clf.fit(x_train, y_train)
    
pred = clf.predict_proba(test)
pred = pred[:, 1]

submission = test[['SK_ID_CURR']]
submission['TARGET'] = pred

if full_dataset:
    submission.to_csv('xgb.csv', index=False)
else:
    submission.to_csv('xgb_fs.csv', index=False)
    
