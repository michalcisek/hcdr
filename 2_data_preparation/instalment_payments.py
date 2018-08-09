# -*- coding: utf-8 -*-
"""
SK_ID_PREV - "ID of previous credit in Home credit related to loan in our sample.
    (One loan in our sample can have 0,1,2 or more previous loans in Home Credit)",hashed
SK_ID_CURR - ID of loan in our sample

NUM_INSTALMENT_VERSION - Version of installment calendar (0 is for credit card) of previous credit.
    Change of installment version from month to month signifies that some parameter of payment calendar has changed
NUM_INSTALMENT_NUMBER - On which installment we observe payment

DAYS_INSTALMENT - When the installment of previous credit was supposed to be paid
    (relative to application date of current loan), time only relative to the application
DAYS_ENTRY_PAYMENT - When was the installments of previous credit paid actually
    (relative to application date of current loan), time only relative to the application

AMT_INSTALMENT - What was the prescribed installment amount of previous credit on this installment
AMT_PAYMENT - What the client actually paid on previous credit on this installment

"""
"""
ideas to improve:
    - handle outliers in payment_delay and payment_diff
"""

import pandas as pd
import numpy as np
import pyarrow.feather as feather
import os

table = "InstPaym"

#get most frequent value for given series
def most_freq_val(x):
    if x.shape[0] == 0:
        res = np.nan
    else:
        try:
            res = x.value_counts().idxmax()
        except ValueError:
            res = np.nan
    return res

#get number of different values for given series
def diff_vals(x):
    dv = x.nunique()
    return dv

ip = pd.read_csv('1_data_import\\installments_payments.csv')

#NUM_INSTALMENT_VERSION - most frequent value, number of different values
num_inst_ver = ip.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].\
    agg([most_freq_val, diff_vals]).\
    rename(columns = {'most_freq_val': 'freq_inst_ver', 'diff_vals': 'diff_inst_ver'}).\
    reset_index()

#NUM_INSTALMENT_NUMBER  - maximum and average of instalments. Indicator of the lenght of loan
num_inst_len = ip.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].\
    agg([np.mean, np.max]).\
    rename(columns = {'mean': 'inst_num_mean', 'amax': 'inst_num_max'}).\
    reset_index()

#calculate payment for every instalment in every loan
ip['payment_delay'] = ip['DAYS_ENTRY_PAYMENT'] - ip['DAYS_INSTALMENT']

#calculate difference between expected and given payment;
#negative values mean that client paid less than it should be paid
ip['payment_diff'] = ip['AMT_PAYMENT'] - ip['AMT_INSTALMENT']

#payment_delay, payment_diff - typical aggregation functions
pmnts = ip.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['payment_delay', 'payment_diff'].\
    agg([np.sum, np.mean, np.std]).\
    reset_index()

pmnts.columns = [i[0] if i[1] == '' else '_'.join(i) for i in pmnts.columns.values]

#join all tables
df = pd.merge(pmnts, num_inst_len, on = ["SK_ID_CURR", "SK_ID_PREV"])
df = pd.merge(df, num_inst_ver, on = ["SK_ID_CURR", "SK_ID_PREV"])

#aggregate by SK_ID_CURR
df = df.drop(['SK_ID_PREV'], axis = 1)

df = df.groupby('SK_ID_CURR').\
    agg([np.mean, np.sum, np.std]).\
    reset_index()

df.columns = [i[0] if i[1] == '' else '_'.join(reversed(i)) for i in df.columns.values]

#save as feather file
#feather.write_feather(df, 'instalment_payments')

suffix = '.feather'

filePath = os.path.join(os.getcwd(), '2_data_preparation', 'features', table + '_features' + suffix)

feather.write_feather(df, filePath)