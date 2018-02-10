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


def transform_data_pipe(data, numeric_columns, categorical_columns):
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

    cleaned_data = pipeline.fit_transform(data)

    return cleaned_data


def random_forest_grid(X, y, params):
    random_forest = RandomForestClassifier()
    grid = GridSearchCV(random_forest, params, scoring='accuracy', cv=10)
    grid_fit = grid.fit(X, y)
    return grid_fit.best_params_


X = read_data(TRAIN_FILE_PATH)
X.dropna(subset=['Embarked'], inplace=True)
y = X.Survived  # get the response variable
# y = X.Survived.apply(lambda x: 'Yes' if x == 1 else 'No')  # get the response variable
# name: not important
# Ticket: remove for now, it's very inconsistent. Don't know what to do
# Cabin:  Remove, only first class passengers have cabin
columns_to_remove = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId']
cat_columns = ['Embarked', 'Sex', 'Pclass']
X = remove_columns(X, columns_to_remove)
X_labels = X.columns
X_numeric = remove_columns(X, cat_columns)
num_columns = list(X_numeric)

param_grid = [
    {'n_estimators': [5, 10, 15, 20, 25], 'max_features': [2, 4, 6, 8, 10]},
    {'bootstrap': [False], 'n_estimators': [5, 10, 15, 20, 25], 'max_features': [2, 4, 6, 8, 10]},
]

transformed_data = transform_data_pipe(X, num_columns, cat_columns)
rf_params = random_forest_grid(transformed_data, y, param_grid)

random_for = RandomForestClassifier(max_features=rf_params['max_features'], n_estimators=rf_params['n_estimators'],
                                    random_state=RANDOM_SEED)
random_for.fit(transformed_data, y)

# random_class = RandomForestClassifier(max_features=8, n_estimators=20, random_state=RANDOM_SEED)
# fit = random_class.fit(cleaned_data, y)
# score = cross_val_score(fit, cleaned_data, y, cv=10)

# predict
X_test = read_data(TEST_FILE_PATH)
passenger_id = X_test.PassengerId

X_test = remove_columns(X_test, columns_to_remove[1:-1])
transformed_test_data = transform_data_pipe(X_test, num_columns, cat_columns)
predicted = random_for.predict(transformed_test_data)
predicted_series = pd.Series(predicted)

df = pd.DataFrame()
df['PassengerId'] = passenger_id
df['Survived'] = predicted_series
df.to_csv('results.csv', columns=['PassengerId', 'Survived'], index=False)
print('mgg')