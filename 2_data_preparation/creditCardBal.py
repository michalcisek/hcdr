# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:10:49 2018

@author: klenczewsk001

-- pyt co to jest AMT_BALANCE - zgodnie z internetem to 
mowi to o wykorzystaniu dostepnego limitu, czyli ile było pociagniec w danym miesiacu
-- pokazac ze nie ma sensu podwojne grupowanie, zazwyczaj tylko 1 prev kred per curr_id
-- co oznaczaja te wszystkie charakterystyki dot receivable
"""

import pandas as pd
import pyarrow.feather as feather
import gc

table = 'credCardBal'
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

credCardBal['ReceiveablePrincipalToTotal'] = credCardBal.AMT_RECEIVABLE_PRINCIPAL/credCardBal.AMT_TOTAL_RECEIVABLE

credCardBal['TotalPaymentToMinInst'] = credCardBal.AMT_PAYMENT_TOTAL_CURRENT/credCardBal.AMT_INST_MIN_REGULARITY

credCardBal['AmtPerATMDrawing'] = credCardBal.AMT_DRAWINGS_ATM_CURRENT/credCardBal.CNT_DRAWINGS_ATM_CURRENT

credCardBal['AmtPerCurrentDrawing'] = credCardBal.AMT_DRAWINGS_CURRENT/credCardBal.CNT_DRAWINGS_CURRENT
credCardBal['AmtPerOtherDrawing'] = credCardBal.AMT_DRAWINGS_OTHER_CURRENT/credCardBal.CNT_DRAWINGS_OTHER_CURRENT
credCardBal['AmtPerPOSDrawing'] = credCardBal.AMT_DRAWINGS_POS_CURRENT/credCardBal.CNT_DRAWINGS_POS_CURRENT

agg = credCardBal.drop(columns = ['SK_ID_PREV', 'CNT_INSTALMENT_MATURE_CUM']).groupby(['SK_ID_CURR']).agg(['mean', 'min', 'max','sum','var','std'])

columns = []
for var in agg.columns.levels[0]:
    for stat in agg.columns.levels[1]:
        columns.append('%s_%s_%s' % (table, var, stat))
        
agg.columns = columns
agg['SK_ID_CURR'] = agg.index

application = application.merge(agg, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del table, app_train, app_test, credCardBal, df
gc.collect()

feather.write_feather(application, 'credCardBal_features')
