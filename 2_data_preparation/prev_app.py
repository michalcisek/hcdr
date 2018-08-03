'''
TODO: WRITE COMMENTS HERE
    - skrocic skrypt
    - opakowac w fcje
    - 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pyarrow.feather as feather
import gc
import functions as f

table = 'prev_app'

#os.getcwd()
#path = "C:\\Users\klenczewsk001\Desktop\kaggle_HomeCredit\hcdr\hcdr"
#os.chdir(path)
app_train = pd.read_csv(r'1_data_import\application_train.csv')
app_test = pd.read_csv(r'1_data_import\application_test.csv')
prev_app = pd.read_csv(r'1_data_import\previous_application.csv')
df = app_train.append(app_test).reset_index()
#df to be joined
application = df[['SK_ID_CURR']]
#get rid of system errors
prev_app = prev_app[prev_app.FLAG_LAST_APPL_PER_CONTRACT != 'N']
prev_app = prev_app[prev_app.NFLAG_LAST_APPL_IN_DAY != 0]
# get rid of columns helping identified system errors
columns = ['FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY']
prev_app = prev_app.drop(columns, axis = 1)

numericCols = ['SK_ID_CURR', 'AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_DOWN_PAYMENT','AMT_GOODS_PRICE','CNT_PAYMENT','DAYS_DECISION','HOUR_APPR_PROCESS_START',
'RATE_DOWN_PAYMENT','DAYS_TERMINATION','DAYS_LAST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_FIRST_DUE','DAYS_FIRST_DRAWING']
dfNumerics = prev_app[numericCols]
application = f.agg_numeric(dfNumerics, 'SK_ID_CURR', table, application)

objectCols = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START','CODE_REJECT_REASON','NAME_TYPE_SUITE',
            'NAME_PAYMENT_TYPE','NAME_CASH_LOAN_PURPOSE']
dfObjects = prev_app[objectCols]
application = count_categorical(dfObjects, 'SK_ID_CURR', 'prev_app', application)

# create features representing statistics for Approved and Canceled applications
Approved = [1 if i == 'Approved' else 0 for i in prev_app.NAME_CONTRACT_STATUS]
Canceled = [1 if i == 'Canceled' else 0 for i in prev_app.NAME_CONTRACT_STATUS]

#df representing values for approved loans
#set(prev_app.dtypes.values)
numerics = ['int64', 'float64']
prev_app_Approved = prev_app.select_dtypes(include = numerics).drop(columns = ['SK_ID_PREV'])
#df.iloc[:,1:].div(df.A, axis=0)
prev_app_Approved = pd.concat([prev_app_Approved.SK_ID_CURR, prev_app_Approved.iloc[:, 1:].multiply(Approved, axis = 0)], axis = 1)

application = f.agg_numeric(prev_app_Approved, 'SK_ID_CURR', 'prevApp', application)

#df representing values for cancelled loans
prev_app_Canceled = prev_app.select_dtypes(include = numerics).drop(columns = ['SK_ID_PREV'])
prev_app_Canceled = pd.concat([prev_app_Canceled.SK_ID_CURR, prev_app_Canceled.iloc[:, 1:].multiply(Canceled, axis = 0)], axis = 1)
application = f.agg_numeric(prev_app_Canceled, 'SK_ID_CURR', 'prevApp', application)

gc.enable()
del prev_app_Approved, prev_app_Canceled, numerics, Approved, Canceled, dfObjects, objectCols
gc.collect()

# save features to feather file
feather.write_feather(application, 'prevApp_features')

