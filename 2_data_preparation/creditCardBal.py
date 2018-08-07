# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:10:49 2018

@author: klenczewsk001
"""

import pandas as pd
import numpy as np
import pyarrow.feather as feather
import gc

table = ''
app_train = pd.read_csv(r'1_data_import\application_train.csv')
app_test = pd.read_csv(r'1_data_import\application_test.csv')
credCardBal = pd.read_csv(r'1_data_import\credit_card_balance.csv')
df = app_train.append(app_test).reset_index()

application = df[['SK_ID_CURR']]

credCardBal = credCardBal[credCardBal.NAME_CONTRACT_STATUS == 'Active']

credCardBal = credCardBal.drop('NAME_CONTRACT_STATUS', 1)

credCardBal['BalanceToLimit'] = credCardBal['AMT_BALANCE']/credCardBal['AMT_CREDIT_LIMIT_ACTUAL']

credCardBal['DrawingsToLimit'] = (credCardBal.AMT_DRAWINGS_ATM_CURRENT + 
           credCardBal.AMT_DRAWINGS_CURRENT + credCardBal.AMT_DRAWINGS_OTHER_CURRENT + 
           credCardBal.AMT_DRAWINGS_POS_CURRENT)/credCardBal.AMT_CREDIT_LIMIT_ACTUAL
           
credCardBal['MinInstToLimit'] = credCardBal.AMT_INST_MIN_REGULARITY/credCardBal.AMT_CREDIT_LIMIT_ACTUAL


credCardBal.head()
