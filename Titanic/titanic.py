# Kaggle's titanic competition --> https://www.kaggle.com/c/titanic

import os
import pandas as pd
import matplotlib.pyplot as plt
from Titanic.dataframe_selector import DataFrameSelector
from Titanic.CategoricalEncoder import CategoricalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

TRAIN_FILE_PATH = './data/train_titanic.csv'
TEST_FILE_PATH = './data/test_titanic.csv'
RANDOM_SEED = 42


def read_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("The file you are trying to access does not exist")
    return pd.read_csv(path)


def remove_columns(data, column_names):
    df = data.drop(column_names, axis=1)
    return df


def plot_histogram(data):
    data.hist(bins=50, figsize=(20, 15))
    plt.show()


def fill_missing_with_median(data, column=None):
    imputer = Imputer(strategy='median')

    if not column:
        imputer.fit(data)
        new_data = imputer.transform(data)
        return pd.DataFrame(new_data, columns=data.columns)
    else:
        temp_data = data[column].values.reshape(-1, 1)
        imputer.fit(temp_data)
        new_data = imputer.transform(temp_data)
        new_data_df = pd.DataFrame(new_data, columns=[column])
        data[column] = new_data_df
        return data


# def encode_categorical(data, column):
#     label_encoder = LabelBinarizer()
#     encoded_column = label_encoder.fit_transform(data[column])
#     data[column] = encoded_column
#     return encoded_column


X = read_data(TRAIN_FILE_PATH)
X.dropna(subset=['Embarked'], inplace=True)
# y = 'Yes' if X.Survived.all() == 1 else 'No'  # get the response variable
y = X.Survived.apply(lambda x: 'Yes' if x == 1 else 'No')  # get the response variable
# name: not important
# Ticket: remove for now, it's very inconsistent. Don't know what to do
# Cabin:  Remove, only first class passengers have cabin
columns_to_remove = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId']
categorical_columns = ['Embarked', 'Sex', 'Pclass']
X = remove_columns(X, columns_to_remove)
X_labels = X.columns
X_numeric = remove_columns(X, categorical_columns)
numeric_columns = list(X_numeric)

numeric_pipeline = Pipeline([
    ('selector', DataFrameSelector(numeric_columns)),
    ('imputer', Imputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('selector', DataFrameSelector(categorical_columns)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
])

pipeline = FeatureUnion(transformer_list=[
    ('numeric_pipeline', numeric_pipeline),
    ('categorical_pipeline', categorical_pipeline),
])

cleaned_data = pipeline.fit_transform(X)
# fill_missing_with_median(X, 'Age')  # fill age with the median value of the column
# ds = DataFrameSelector(categorical_columns)
# X = ds.fit_transform(X)

# random_class = RandomForestClassifier()
# fit = random_class.fit(cleaned_data, y)
# score = cross_val_score(fit, cleaned_data, y, cv=10)

# param_grid = [
#     {'n_estimators': [5, 10, 15, 20, 25], 'max_features': [2, 4, 6, 8, 10]},
#     {'bootstrap': [False], 'n_estimators': [5, 10, 15, 20, 25], 'max_features': [2, 4, 6, 8, 10]},
# ]
#
# grid = GridSearchCV(random_class, param_grid, scoring='accuracy', cv=10)
# grid.fit(cleaned_data, y)
# print(grid.best_params_)

random_class = RandomForestClassifier(max_features=8, n_estimators=20, random_state=RANDOM_SEED)
fit = random_class.fit(cleaned_data, y)
score = cross_val_score(fit, cleaned_data, y, cv=10)


#predict

