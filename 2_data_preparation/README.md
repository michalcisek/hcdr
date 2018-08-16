# data preparation

Pipeline:
- run feature prep scripts
- run mergeFeaturesDF.py - to get dataframe with all features
- run train_test_split.py - to get train and test samples (from abovementioned dataframe), ready to modelling
- run feature_selection.py - to get train and test samples with top 20 most important variables
