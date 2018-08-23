# coding: utf-8

# In[6]:

import numpy as np
import pandas as pd
from pyarrow.feather import write_feather
import warnings
from functools import reduce

warnings.filterwarnings('ignore')


# one-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# In[7]:

app_train = pd.read_csv("1_data_import/application_train.csv")
print('Training data shape: ', app_train.shape)
# app_train.head()

# In[8]:

bureau = pd.read_csv("1_data_import/bureau.csv")
print('Bureau shape: ', bureau.shape)
# bureau.head()

# In[9]:

bureau["CREDIT_ACTIVE_NUM"] = np.where(bureau["CREDIT_ACTIVE"] == "Closed", 0, 1)
bureau.sort_values(by=["SK_ID_CURR", "SK_ID_BUREAU"])

# In[10]:

# bureau_sample = bureau.head(100000)
bureau_max = bureau.groupby("SK_ID_CURR").max()
bureau_max.columns = ["ALL_MAX_" + x for x in bureau_max.columns]
bureau_count = bureau.groupby("SK_ID_CURR").count()
bureau_count.columns = ["ALL_COUNT_" + x for x in bureau_count.columns]
bureau_mean = bureau.groupby("SK_ID_CURR").mean()
bureau_mean.columns = ["ALL_MEAN_" + x for x in bureau_mean.columns]
bureau_sum = bureau.groupby("SK_ID_CURR").sum()
bureau_sum.columns = ["ALL_SUM_" + x for x in bureau_sum.columns]

# In[11]:

bureau_max = bureau_max.reset_index()
bureau_count = bureau_count.reset_index()
bureau_mean = bureau_mean.reset_index()
bureau_sum = bureau_sum.reset_index()

# In[ ]:


# In[12]:

bureau_balance = pd.read_csv("1_data_import/bureau_balance.csv")
bureau_balance_last_active = bureau_balance[bureau_balance.STATUS != 'C'].groupby('SK_ID_BUREAU').max().reset_index()

# In[13]:

bureau_from_last_12M = pd.merge(bureau
                                , bureau_balance_last_active
                                , how="left"
                                , on="SK_ID_BUREAU"
                                )
bureau_from_last_12M = bureau_from_last_12M[bureau_from_last_12M.MONTHS_BALANCE >= -12]

# In[14]:

# bureau_sample = bureau.head(100000)
bureau_max_from_last_12M = bureau_from_last_12M.groupby("SK_ID_CURR").max()
bureau_max_from_last_12M.columns = ["12M_MAX_" + x for x in bureau_max_from_last_12M.columns]
bureau_count_from_last_12M = bureau_from_last_12M.groupby("SK_ID_CURR").count()
bureau_count_from_last_12M.columns = ["12M_COUNT_" + x for x in bureau_count_from_last_12M.columns]
bureau_mean_from_last_12M = bureau_from_last_12M.groupby("SK_ID_CURR").mean()
bureau_mean_from_last_12M.columns = ["12M_MEAN_" + x for x in bureau_mean_from_last_12M.columns]
bureau_sum_from_last_12M = bureau_from_last_12M.groupby("SK_ID_CURR").sum()
bureau_sum_from_last_12M.columns = ["12M_SUM_" + x for x in bureau_sum_from_last_12M.columns]

# In[15]:

bureau_max_from_last_12M = bureau_max_from_last_12M.reset_index()
bureau_count_from_last_12M = bureau_count_from_last_12M.reset_index()
bureau_mean_from_last_12M = bureau_mean_from_last_12M.reset_index()
bureau_sum_from_last_12M = bureau_sum_from_last_12M.reset_index()

# In[16]:

dfs = [bureau_max, bureau_count, bureau_max_from_last_12M, bureau_count_from_last_12M, bureau_mean_from_last_12M,
       bureau_sum_from_last_12M, bureau_mean, bureau_sum]

# In[17]:

bureau_final = reduce(lambda left, right: pd.merge(left, right, on='SK_ID_CURR'), dfs)
df = one_hot_encoder(bureau_final, nan_as_category=True)

# In[18]:

# bureau_final.to_csv('bureau_features.csv')
write_feather(df[0], 'bureau_features')
# print("End of script")
