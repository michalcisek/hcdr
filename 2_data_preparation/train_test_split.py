import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy
import pyarrow.feather as feather
import os

os.chdir("C:/Users/ppitera002/Documents/hcdr/hcdr")

filePath = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'sample.feather')
df = feather.read_feather(filePath)

X = df.drop(['TARGET'], axis=1)
y = df["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

TRAIN = pd.concat([X_train, y_train], axis=1)
TEST = pd.concat([X_test, y_test], axis=1)

filePathTRAIN = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TRAIN_sample.feather')
filePathTEST = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TEST_sample.feather')

feather.write_feather(TRAIN, filePathTRAIN)
feather.write_feather(TEST, filePathTEST)
