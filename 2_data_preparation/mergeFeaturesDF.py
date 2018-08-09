# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 19:22:16 2018

@author: klenczewsk001
"""
import glob
import os
import pyarrow.feather as feather
import pandas as pd    
from functools import reduce

filePath = os.path.join(os.getcwd(), '2_data_preparation', 'features')

all_files = glob.glob(os.path.join(filePath, "*.feather"))

dfList = []
for file in all_files:
    file_name = os.path.splitext(os.path.basename(file))[0]
    df = feather.read_feather(file)
    dfList.append(df)
    
featuresDF = reduce(lambda x, y: pd.merge(x, y, on = 'SK_ID_CURR'), dfList)
