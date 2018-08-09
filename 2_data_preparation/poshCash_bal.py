'''
TODO: 
    - na jakich rachunkach liczymy zmienne? tylko ze statusem Active?
    - 
    - 
'''
import os as os
import pandas as pd
import pyarrow.feather as feather

table = 'poshCashBal'
app_train = pd.read_csv(r'1_data_import\application_train.csv')
app_test = pd.read_csv(r'1_data_import\application_test.csv')
poshCash_bal = pd.read_csv(r'1_data_import\POS_CASH_balance.csv')
df = app_train.append(app_test).reset_index()

application = df[['SK_ID_CURR']]

poshCash_bal = poshCash_bal[poshCash_bal.NAME_CONTRACT_STATUS == 'Active']
numerics = ['int64', 'float64']
poshCash_bal = poshCash_bal.select_dtypes(include = numerics).drop(columns = ['MONTHS_BALANCE'])
#spr na 1 ekspozycji
#test_df = poshCash_bal[poshCash_bal.SK_ID_CURR == 334279].select_dtypes(include = ['int64'])

agg1 = poshCash_bal.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg(['mean', 'min', 'max','sum','var','std'])
columns = []
for var in agg1.columns.levels[0]:
    for stat in agg1.columns.levels[1]:
        columns.append('%s_%s_%s' % (table, var, stat))

agg1.columns = columns

agg2 = agg1.groupby(['SK_ID_CURR']).agg(['mean', 'min', 'max','sum','var','std'])

columns = []
for var in agg2.columns.levels[0]:
    for stat in agg2.columns.levels[1]:
        columns.append('%s_%s' % (var, stat))

agg2.columns = columns
agg2['SK_ID_CURR'] = agg2.index

application = application.merge(agg2, on = 'SK_ID_CURR', how = 'left')

suffix = '.feather'

filePath = os.path.join(os.getcwd(), '2_data_preparation', 'features', table + '_features' + suffix)

feather.write_feather(application, filePath)

