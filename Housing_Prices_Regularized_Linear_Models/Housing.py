"""
* Problem: Kaggle's "House Prices: Advanced Regression Techniques" competition
* URL: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
* Goal: It is your job to predict the sales price for each house. For each Id in the test set, you must predict the
  value of the SalePrice variable.
* Metric: Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value
  and the logarithm of the observed sales price.
* Approach: I'm going to tackle this problem using regularized linear models: Ridge, Lasso and Elastic net
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def read_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("The file you are trying to access does not exist")
    return pd.read_csv(path, header=0, )


def remove_columns(data_sets, columns_list):
    for df in data_sets:
        df.drop(columns=columns_list, axis=1, inplace=True)


def fill_missing_values(data_sets, column, strategy='custom', value=None):
    for df in data_sets:
        if df[column].dtype == 'object':
            if strategy == 'mode':
                df[column] = df[column].fillna(df[column].value_counts().idxmax())
            elif strategy == 'custom' and value is not None:
                df[column] = df[column].fillna(value)
        elif df[column].dtype == 'int64' or df[column].dtype == 'float64':
            if strategy == 'mode':
                df[column] = df[column].fillna(df[column].mode())
            elif strategy == 'median' and value is not None:
                df[column] = df[column].fillna(df[column].median())
            elif strategy == 'custom' and value is not None:
                df[column] = df[column].fillna(value)


def create_dummy_variables(data_sets, columns_names):
    for i in range(len(data_sets)):
        data_sets[i] = pd.get_dummies(data_sets[i], columns=columns_names, prefix=columns_names)

def scale_data(data_sets):
    scaler = StandardScaler()
    scaler.fit(data_sets[0] + data_sets[1])

    for df in data_sets:
        scaler.transform(df)


# script constants
RANDOM_SEED = 42
TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'

X = read_data(TRAIN_PATH)
y = X.SalePrice
X.drop('SalePrice', axis=1, inplace=True)
test = read_data(TEST_PATH)
all_data = [X, test]

# columns to remove from model
# dropping the ones that have more than 50% missing values
columns_to_drop = ['Fence', 'Id', 'Alley', 'PoolQC', 'MiscFeature']
remove_columns(all_data, columns_to_drop)

# fill missing values
fill_missing_values(all_data, 'FireplaceQu', strategy='custom', value='None')
fill_missing_values(all_data, 'LotFrontage', strategy='median')
fill_missing_values(all_data, 'MasVnrType', strategy='mode')
fill_missing_values(all_data, 'MasVnrArea', strategy='mode')
fill_missing_values(all_data, 'BsmtQual', strategy='mode')
fill_missing_values(all_data, 'BsmtCond', strategy='mode')
fill_missing_values(all_data, 'BsmtExposure', strategy='mode')
fill_missing_values(all_data, 'BsmtFinType1', strategy='mode')
fill_missing_values(all_data, 'BsmtFinType2', strategy='mode')
fill_missing_values(all_data, 'Electrical', strategy='mode')
fill_missing_values(all_data, 'GarageType', strategy='custom', value='None')
fill_missing_values(all_data, 'GarageYrBlt', strategy='custom', value=0000)  # fill with year 0000
fill_missing_values(all_data, 'GarageFinish', strategy='custom', value='None')
fill_missing_values(all_data, 'GarageQual', strategy='custom', value='None')
fill_missing_values(all_data, 'GarageCond', strategy='custom', value='None')

# One hot encoding
cat_cols = set(X.select_dtypes(include=['object']).columns)
num_cols = set(X.columns) - cat_cols

create_dummy_variables(all_data, list(cat_cols))

#scale predictors
scale_data(all_data)