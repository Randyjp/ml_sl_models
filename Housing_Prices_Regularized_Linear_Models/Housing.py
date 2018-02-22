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
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict



def read_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("The file you are trying to access does not exist")
    return pd.read_csv(path, header=0, )


def remove_columns(data_set, columns_list):
    data_set.drop(columns=columns_list, axis=1, inplace=True)


def fill_missing_values(data_set, column, strategy='custom', value=None):
    if data_set[column].dtype == 'object':
        if strategy == 'mode':
            data_set[column].fillna(data_set[column].value_counts().idxmax(), inplace=True)
        elif strategy == 'custom' and value is not None:
            data_set[column].fillna(value, inplace=True)
    elif data_set[column].dtype == 'int64' or data_set[column].dtype == 'float64':
        if strategy == 'mode':
            data_set[column].fillna(data_set[column].mode(), inplace=True)
        elif strategy == 'median':
            data_set[column].fillna(data_set[column].median(), inplace=True)
        elif strategy == 'custom' and value is not None:
            data_set[column].fillna(value, inplace=True)


def create_dummy_variables(data_set, columns_names):
    return pd.get_dummies(data_set, columns=columns_names, prefix=columns_names, sparse=False)


def scale_data(data_set):
    scaler = StandardScaler()
    scaler.fit(data_set)
    return scaler.transform(data_set)


# script constants
RANDOM_SEED = 42
TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'
RESULT_PATH = './data/result.csv'


X = read_data(TRAIN_PATH)
y = X.SalePrice
X.drop('SalePrice', axis=1, inplace=True)
X_test = read_data(TEST_PATH)
all_data = pd.concat([X, X_test])
ids = all_data.Id
test_ids = X_test.Id
# columns to remove from model
# dropping the ones that have more than 50% missing values
columns_to_drop = ['Fence', 'Id', 'Alley', 'PoolQC', 'MiscFeature']
remove_columns(all_data, columns_to_drop)

# fill missing values
fill_missing_values(all_data, 'FireplaceQu', strategy='custom', value='Nada')
fill_missing_values(all_data, 'LotFrontage', strategy='median')
fill_missing_values(all_data, 'MasVnrType', strategy='mode')
fill_missing_values(all_data, 'MasVnrArea', strategy='custom', value=0.0)  # they have not MasVnr, fill are with 0
fill_missing_values(all_data, 'BsmtQual', strategy='mode')
fill_missing_values(all_data, 'BsmtCond', strategy='mode')
fill_missing_values(all_data, 'BsmtExposure', strategy='mode')
fill_missing_values(all_data, 'BsmtFinType1', strategy='mode')
fill_missing_values(all_data, 'BsmtFinType2', strategy='mode')
fill_missing_values(all_data, 'Electrical', strategy='mode')
fill_missing_values(all_data, 'GarageType', strategy='custom', value='Nada')
fill_missing_values(all_data, 'GarageYrBlt', strategy='custom', value=0000)  # fill with year 0000
fill_missing_values(all_data, 'GarageFinish', strategy='custom', value='Nada')
fill_missing_values(all_data, 'GarageQual', strategy='custom', value='Nada')
fill_missing_values(all_data, 'GarageCond', strategy='custom', value='Nada')
#### test set ###
fill_missing_values(all_data, 'MSZoning', strategy='mode')
fill_missing_values(all_data, 'Functional', strategy='mode')
fill_missing_values(all_data, 'BsmtFinSF2', strategy='custom', value=0)
fill_missing_values(all_data, 'BsmtFinSF1', strategy='custom', value=0)
fill_missing_values(all_data, 'TotalBsmtSF', strategy='custom', value=0)
fill_missing_values(all_data, 'BsmtFullBath', strategy='custom', value=0)
fill_missing_values(all_data, 'BsmtHalfBath', strategy='custom', value=0)
fill_missing_values(all_data, 'BsmtUnfSF', strategy='custom', value=0)
fill_missing_values(all_data, 'GarageArea', strategy='custom', value=0)
fill_missing_values(all_data, 'GarageCars', strategy='custom', value=0)
fill_missing_values(all_data, 'Utilities', strategy='mode')
fill_missing_values(all_data, 'SaleType', strategy='mode')
fill_missing_values(all_data, 'Exterior2nd', strategy='mode')
fill_missing_values(all_data, 'Exterior1st', strategy='mode')
fill_missing_values(all_data, 'KitchenQual', strategy='mode')

# One hot encoding
cat_cols = set(all_data.select_dtypes(include=['object']).columns)
num_cols = set(all_data.columns) - cat_cols

all_data = create_dummy_variables(all_data, list(cat_cols))
all_cols = list(all_data.columns)
# scale predictors
all_data = pd.DataFrame(data=scale_data(all_data), columns=all_cols, index=ids)
X = all_data[0:1460]
X_test = all_data[1460:]
all_data = None

# Lasso
lasso = LassoCV(cv=10, random_state=RANDOM_SEED)
lasso_fitted = lasso.fit(X, y)
lasso_score = lasso.score(X, y)

# ridge
ridge = LassoCV(cv=10, random_state=RANDOM_SEED)
ridge_fitted = ridge.fit(X, y)
ridge_score = ridge.score(X, y)

# elastic ne
elastic = ElasticNetCV(cv=10, random_state=RANDOM_SEED)
elastic_fitted = elastic.fit(X, y)
elastic_score = ridge.score(X, y)


#predict
predicted = lasso_fitted.predict(X_test)
result = pd.DataFrame()
result['Id'] = test_ids
result['SalePrice'] = predicted
result.to_csv(RESULT_PATH, columns=['Id', 'SalePrice'], index=False)
