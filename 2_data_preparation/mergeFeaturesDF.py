# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 19:22:16 2018

-- dorzucic train/test z Targetem

@author: mcisek001
"""

import glob
import os
import pyarrow.feather as feather

import pandas as pd    
from functools import reduce


filePath = os.path.join(os.getcwd(), '2_data_preparation', 'features')

all_files = glob.glob(os.path.join(filePath, "*.feather"))

#dfList = []
#for file in all_files:
#    file_name = os.path.splitext(os.path.basename(file))[0]
#    df = feather.read_feather(file)
#    dfList.append(df)
    
featuresDF = feather.read_feather(all_files[0])

for file in all_files[1:]:
    add_df = feather.read_feather(file)
    
    #remove the same columns from second data frame
    cols_x = featuresDF.columns
    cols_y = add_df.columns
    to_remove = list(set(cols_x) & set(cols_y))
    to_remove.remove('SK_ID_CURR')
    add_df.drop(columns=to_remove, inplace=True)
    
    featuresDF = pd.merge(featuresDF, add_df, how = 'left', on = 'SK_ID_CURR')    
        
#featuresDF = reduce(lambda x, y: pd.merge(x, y, on = 'SK_ID_CURR'), dfList)

filePath = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'sample.feather')

feather.write_feather(featuresDF, filePath)