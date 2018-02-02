# Naive Bayes(I'll be using Gaussian distributions)

# Things to keep in mind:
# 1)	Assumes independent features
# 2)	Useful for large number of predictors(p)
# 3)	Assumes each class’s covariance matrix is diagonal
# 4)	Can handle qualitative and quantitative features

# Note: I'll use this code to experiment with GaussianNB and the effects of standardizing the data

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 42  # the meaning of life

X, y = load_breast_cancer(return_X_y=True)  # returns X and y instead of an object

# split the data(35% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size=.35)

# fit, predict and test without scaling
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predicted = nb.predict(X_test)
accu_score = accuracy_score(y_test, nb_predicted)
matrix = confusion_matrix(y_test, nb_predicted)
print(accu_score)
print(matrix)


# fit, predict and test with scaling
std_nb = make_pipeline(StandardScaler(), GaussianNB())
std_nb.fit(X_train, y_train)
nb_std_predicted = std_nb.predict(X_test)
std_accu_score = accuracy_score(y_test, nb_std_predicted)
matrix_std = confusion_matrix(y_test, nb_std_predicted)
print(std_accu_score)
print(matrix_std)

