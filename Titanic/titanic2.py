# Kaggle's titanic competition --> https://www.kaggle.com/c/titanic
# approach after exploring the discussion section
# following this notebook --> https://www.kaggle.com/sinakhorami/titanic-best-working-classifier

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    return pd.read_csv(path, header=0, )


def remove_columns(data, column_names):
    df = data.drop(column_names, axis=1, inplace=True)
    return df


def plot_histogram(data):
    data.hist(bins=50, figsize=(20, 15))
    plt.show()


def check_feature_vs_target(data, feature, target='Survived'):
    impact = (data[[feature, target]].groupby([feature], as_index=False).mean())
    return impact


def transform_data_pipe(data, numeric_columns, categorical_columns):
    numeric_pipeline = Pipeline([
        ('selector', DataFrameSelector(numeric_columns)),
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
    grid = GridSearchCV(random_forest, params, scoring='f1', cv=10)
    grid_fit = grid.fit(X, y)
    return grid_fit.best_params_


def get_title(name):
    # after the first space, look for a alphabetic string until you find a dot.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    return title_search.group(1) if title_search else ""


X = read_data(TRAIN_FILE_PATH)
X_test = read_data(TEST_FILE_PATH)
passenger_id = X_test.PassengerId
all_data = [X, X_test]  # array with both ds, convenient to apply transformations to data.

print(X.info())  # See all features, check type and if there's missing values.

for col in X.columns[2:]:  # print all vars vs survived mean
    print(check_feature_vs_target(X, col))

for dataset in all_data:
    # create a new feature called family size = #siblings/spouse + #children/parents + 1(person)
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in all_data:
    # create the is Alone feature
    dataset['IsAlone'] = 0
    # select all in which familtsize = 1, set IsAlone to 1
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# count number most common 'Embarked' occurrence and fill NA with it
embarked_count = X['Embarked'].value_counts()
# print(embarked_count)
for dataset in all_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# fill missing values of Fare/age using median
for dataset in all_data:
    dataset['Fare'] = dataset['Fare'].fillna(X['Fare'].median())
    dataset['Age'] = dataset['Age'].fillna(X['Age'].median())

for dataset in all_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in all_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

columns_to_remove = ['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch']

for dataset in all_data:
    remove_columns(dataset, columns_to_remove)

y = X.Survived
X.drop(columns=['Survived'], inplace=True)

#######

cat_columns = ['Embarked', 'Sex', 'Pclass', 'IsAlone', 'Title']
num_columns = ['Age', 'Fare', 'FamilySize']
transformed_data = transform_data_pipe(X, num_columns, cat_columns)
transformed_data_test = transform_data_pipe(X_test, num_columns, cat_columns)
########

param_grid = [
    {'n_estimators': [5, 10, 15, 20, 25], 'max_features': [2, 4, 6, 8, 10, 12, 14, 16, 18]},
    {'bootstrap': [False], 'n_estimators': [5, 10, 15, 20, 25], 'max_features': [2, 4, 6, 8, 10, 12, 14, 16, 18]},
]

rf_params = random_forest_grid(transformed_data, y, param_grid)

rf_model = RandomForestClassifier(max_features=rf_params['max_features'], n_estimators=rf_params['n_estimators'],
                                  random_state=RANDOM_SEED)
rf_model.fit(transformed_data, y)

# predict
predicted = rf_model.predict(transformed_data_test)
predicted_series = pd.Series(predicted)

df = pd.DataFrame()
df['PassengerId'] = passenger_id
df['Survived'] = predicted_series
df.to_csv('results2.csv', columns=['PassengerId', 'Survived'], index=False)
print('Done')


#
# X.dropna(subset=['Embarked'], inplace=True)
# y = X.Survived  # get the response variable
# # y = X.Survived.apply(lambda x: 'Yes' if x == 1 else 'No')  # get the response variable
# # name: not important
# # Ticket: remove for now, it's very inconsistent. Don't know what to do
# # Cabin:  Remove, only first class passengers have cabin
# columns_to_remove = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId']
# cat_columns = ['Embarked', 'Sex', 'Pclass']
# X = remove_columns(X, columns_to_remove)
# X_labels = X.columns
# X_numeric = remove_columns(X, cat_columns)
# num_columns = list(X_numeric)
#
# param_grid = [
#     {'n_estimators': [5, 10, 15, 20, 25], 'max_features': [2, 4, 6, 8, 10]},
#     {'bootstrap': [False], 'n_estimators': [5, 10, 15, 20, 25], 'max_features': [2, 4, 6, 8, 10]},
# ]
#
# transformed_data = transform_data_pipe(X, num_columns, cat_columns)
# rf_params = random_forest_grid(transformed_data, y, param_grid)
#
# random_for = RandomForestClassifier(max_features=rf_params['max_features'], n_estimators=rf_params['n_estimators'],
#                                     random_state=RANDOM_SEED)
# random_for.fit(transformed_data, y)
#
# # random_class = RandomForestClassifier(max_features=8, n_estimators=20, random_state=RANDOM_SEED)
# # fit = random_class.fit(cleaned_data, y)
# # score = cross_val_score(fit, cleaned_data, y, cv=10)
#
# # predict
# passenger_id = X_test.PassengerId
#
# X_test = remove_columns(X_test, columns_to_remove[1:-1])
# transformed_test_data = transform_data_pipe(X_test, num_columns, cat_columns)
# predicted = random_for.predict(transformed_test_data)
# predicted_series = pd.Series(predicted)
#
# df = pd.DataFrame()
# df['PassengerId'] = passenger_id
# df['Survived'] = predicted_series
# df.to_csv('results.csv', columns=['PassengerId', 'Survived'], index=False)
