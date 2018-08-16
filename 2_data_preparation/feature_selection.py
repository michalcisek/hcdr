

# Feature Importance with Extra Trees Classifier
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import os
import pyarrow.feather as feather

# os.chdir("C:/Users/ppitera002/Documents/hcdr/hcdr")

filePathTRAIN = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TRAIN_sample.feather')
filePathTEST = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TEST_sample.feather')
df = feather.read_feather(filePathTRAIN)
df1 = df.drop("index",axis=1)
df1 = df1.fillna(0)
colnames = df1.columns.values
dfY = df1["TARGET"]
Y = dfY.values
dfX = df1.drop(["TARGET"], axis=1)
X = dfX.values
model = ExtraTreesClassifier()
model.fit(X, Y)
# print(model.feature_importances_)
list_imp = model.feature_importances_
top_values = sorted(list_imp, key = lambda x: -x)[0:20]
top_ind = [i for i, j in enumerate(list_imp) if j in top_values]
top_colnames = [j for i, j in enumerate(colnames) if i in top_ind]

test_df = feather.read_feather(filePathTEST)
TRAIN_FS_df = pd.concat([df["TARGET"], df[top_colnames]], axis=1)
TEST_FS_df = pd.concat([test_df["TARGET"], test_df[top_colnames]])
print(TRAIN_FS_df.shape, TEST_FS_df.shape)

filePathTRAIN = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TRAIN_sample_afterFS.feather')
feather.write_feather(TRAIN_FS_df, filePathTRAIN)
filePathTEST = os.path.join(os.getcwd(), '2_data_preparation', 'features', 'TEST_sample_afterFS.feather')
feather.write_feather(TEST_FS_df, filePathTEST)
