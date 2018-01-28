from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


# load the data set we'll be working with. In this case the Boston housing
boston = load_boston()
y = boston.target  # response variable(y) = median house price

# Task 1) make a linear regression model with lstat to predict median value
lsat = boston.data[:, 12].reshape(-1, 1)  # select the 13th column and make it matrix(reshape)
lr1 = LinearRegression()  # create the object
lr1.fit(lsat, y)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr1, lsat, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))  # predicted values
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # regression line
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
# uncomment line below to show graph
plt.show()
