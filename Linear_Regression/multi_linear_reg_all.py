from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd


# load the data set we'll be working with. In this case the Boston housing
boston = load_boston()
boston_df = pd.DataFrame(data=boston.data, columns=boston.feature_names) # get it into a pandas data frame
y = pd.DataFrame(data=boston.data) # get it into a pandas data frame
# boston_df.describe() # take a look at the data
boston = None # help garbage collector

# Task 3) make a linear regression model with all inputs to predict median value
lr1 = LinearRegression()  # create the object
lr1.fit(boston_df, y)
# print(lr1.coef_)
# print(lr1.intercept_)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr1, boston_df, y, cv=10)
scores = cross_val_score(lr1, boston_df, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))  # predicted values
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # regression line
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
# uncomment line below to show graph
plt.show()
