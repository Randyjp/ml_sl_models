from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd

# load the data set we'll be working with. In this case the Boston housing
boston = load_boston()
boston_df = pd.DataFrame(data=boston.data, columns=boston.feature_names)  # get it into a pandas data frame
y = pd.DataFrame(data=boston.data)  # get it into a pandas data frame
X = boston_df['LSTAT'].reshape(-1, 1)  # predictor variable
# boston_df.describe() # take a look at the data
boston = None  # help garbage collector

# Task 4) make a polynomial linear regression model with LSTAT and LSTAT^2 to predict median value
poly_features = PolynomialFeatures(degree=2, include_bias=False) # set the degree of new p's = 2
poly_X = poly_features.fit_transform(X) # creates the LSTAT^2

lr1 = LinearRegression()  # create the object
lr1.fit(poly_X, y)
# print(lr1.coef_)
# print(lr1.intercept_)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr1, X, y, cv=10)
scores = cross_val_score(lr1, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))  # predicted values
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # regression line
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
# uncomment line below to show graph
plt.show()
