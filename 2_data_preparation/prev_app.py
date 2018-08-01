'''
TODO: WRITE COMMENTS HERE
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

table = 'prev_app'

#os.getcwd()
#path = "C:\\Users\klenczewsk001\Desktop\kaggle_HomeCredit\hcdr\hcdr"
#os.chdir(path)
app_train = pd.read_csv(r'1_data_import\application_train.csv')
#app_train.head()
prev_app = pd.read_csv('1_data_import\previous_application.csv')
#prev_app.head()

#get rid of system errors

prev_app = prev_app[prev_app.FLAG_LAST_APPL_PER_CONTRACT != 'N']
prev_app = prev_app[prev_app.NFLAG_LAST_APPL_IN_DAY != 0]

# get rid of columns helping identified system errors
columns = ['FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY']
prev_app = prev_app.drop(columns, axis = 1)

# functions used in script
rangeFunc = lambda x: x.max() - x.min()
#get number of different values for given series
def diff_vals(x):
    dv = x.nunique()
    return dv
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

PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var', 'std']:
    for select in ['AMT_ANNUITY',
                   'AMT_APPLICATION',
                   'AMT_CREDIT',
                   'AMT_DOWN_PAYMENT',
                   'AMT_GOODS_PRICE',
                   'CNT_PAYMENT',
                   'DAYS_DECISION',
                   'HOUR_APPR_PROCESS_START',
                   'RATE_DOWN_PAYMENT',
                   'DAYS_TERMINATION',
                   'DAYS_LAST_DUE',
                   'DAYS_LAST_DUE_1ST_VERSION',
                   'DAYS_FIRST_DUE',
                   'DAYS_FIRST_DRAWING'
                   ]:
        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]

application = app_train[['SK_ID_CURR']]
groupby_aggregate_names = []
groupby_cols = PREVIOUS_APPLICATION_AGGREGATION_RECIPIES[0][0]
specs = PREVIOUS_APPLICATION_AGGREGATION_RECIPIES[0][1]
group_object = prev_app.groupby(groupby_cols)
for select, agg in specs:
    groupby_aggregate_name = table + '_' + agg + '_' + select
    application = application.merge(group_object[select]
                          .agg(agg)
                          .reset_index()
                          .rename(index=str,
                                  columns={select: groupby_aggregate_name})
                          [groupby_cols + [groupby_aggregate_name]],
                          on = groupby_cols,
                          how = 'left')
    groupby_aggregate_names.append(groupby_aggregate_name)

#application.head()
# applying range function
PREVIOUS_APPLICATION_AGGREGATION_RANGE = []
for col in ['AMT_ANNUITY',
            'AMT_APPLICATION',
            'AMT_CREDIT',
            'AMT_DOWN_PAYMENT',
            'AMT_GOODS_PRICE',
            'CNT_PAYMENT',
            'DAYS_DECISION',
            'HOUR_APPR_PROCESS_START',
            'RATE_DOWN_PAYMENT',
            'DAYS_TERMINATION',
            'DAYS_LAST_DUE',
            'DAYS_LAST_DUE_1ST_VERSION',
            'DAYS_FIRST_DUE',
            'DAYS_FIRST_DRAWING'
           ]:
        PREVIOUS_APPLICATION_AGGREGATION_RANGE.append((col, 'rangeFunc'))

groupby_aggregate_names = []

group_object = prev_app.groupby(groupby_cols)
for select, agg in PREVIOUS_APPLICATION_AGGREGATION_RANGE:
    groupby_aggregate_name = table + '_' + agg + '_' + select
    application = application.merge(group_object[select]
                          .agg(rangeFunc)
                          .reset_index()
                          .rename(index=str,
                                  columns={select: groupby_aggregate_name})
                          [groupby_cols + [groupby_aggregate_name]],
                          on = groupby_cols,
                          how = 'left')
    groupby_aggregate_names.append(groupby_aggregate_name)
groupby_aggregate_names


# add feature representing number of prev app per SK_ID_CURR
df_toMerge = group_object['SK_ID_PREV'].nunique().reset_index().rename(index=str,columns={select: groupby_aggregate_name})
cols = ['SK_ID_CURR', 'Cnt_prevApp']
df_toMerge.columns = cols
application = application.merge(df_toMerge,
                          on = groupby_cols,
                          how = 'left')

application.head()

PREVIOUS_APPLICATION_AGGREGATION_cntUNIQUE = []
for col in ['NAME_CONTRACT_TYPE',
            'WEEKDAY_APPR_PROCESS_START',
            'NAME_CASH_LOAN_PURPOSE',
            'NAME_PAYMENT_TYPE',
            'CODE_REJECT_REASON',
            'NAME_TYPE_SUITE',
            'NAME_PAYMENT_TYPE',
            'NAME_CASH_LOAN_PURPOSE',
            'SK_ID_PREV'
           ]:
        PREVIOUS_APPLICATION_AGGREGATION_cntUNIQUE.append((col, 'cntU'))
groupby_aggregate_names = []

for select, agg in PREVIOUS_APPLICATION_AGGREGATION_cntUNIQUE:
    groupby_aggregate_name = table + '_' + agg + '_' + select
    application = application.merge(group_object[select]
                          .agg(diff_vals)
                          .reset_index()
                          .rename(index = str,
                                  columns = {select: groupby_aggregate_name})
                          [groupby_cols + [groupby_aggregate_name]],
                          on = groupby_cols,
                          how = 'left')
    groupby_aggregate_names.append(groupby_aggregate_name)
application.head()

# create features representing statistics per 
#prev_app.NAME_CONTRACT_STATUS.unique()
#pd.value_counts(prev_app.NAME_CONTRACT_STATUS)
Approved = [1 if i == 'Approved' else 0 for i in prev_app.NAME_CONTRACT_STATUS]
Canceled = [1 if i == 'Canceled' else 0 for i in prev_app.NAME_CONTRACT_STATUS]

def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. ##TODO uzupelnic
            The columns are also renamed to keep track of features created.
    
    """
    
    # First calculate counts
    counts = pd.DataFrame(df.groupby(group_var, as_index = False)[df.columns[1]].count()).rename(columns = {df.columns[1]: '%s_counts' % df_name})
    
    # Group by the specified variable and calculate the statistics
    agg = df.groupby(group_var).agg(['mean', 'max', 'min', 'sum']).reset_index()
    
    # Need to create new column names
    columns = [group_var]
    
    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
              
    #  Rename the columns
    agg.columns = columns
    
    # Merge with the counts
    agg = agg.merge(counts, on = group_var, how = 'left')
    
    return agg

#df representing values for approved loans
#set(prev_app.dtypes.values)
numerics = ['int64', 'float64']
prev_app_Approved = prev_app.select_dtypes(include = numerics).drop(columns = ['SK_ID_PREV'])
#df.iloc[:,1:].div(df.A, axis=0)
prev_app_Approved = pd.concat([prev_app_Approved.SK_ID_CURR, prev_app_Approved.iloc[:, 1:].multiply(Approved, axis = 0)], axis = 1)

#df representing values for cancelled loans