# Linear Discriminant analysis

# When LDA|QDA perform the best?
#
# 1)	The classes are well separated
# 2)	If n (# of samples) is small and the distribution of the predictors in X is approximately normal.
# 3)	For more than two response classes since it provides low-dimensional views of the data.


from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

wine = load_wine()
X = wine.data
y = wine.target

# fit the model
lda = LinearDiscriminantAnalysis()
lda_fit = lda.fit(X, y)
# print(lda_fit)

# compute accuracy metrics, keep in mind this is ONLY the training error. We might be over|under fitting

# confusion matrix
predicted_cv = cross_val_predict(lda_fit, X, y, cv=10)
con_matrix = confusion_matrix(y, predicted_cv)
# print(con_matrix)  # [[58  1  0] [ 3 66  2] [ 0  0 48]]

# Precision, Recall and F metric
# micro-averaging ==> considering each element of the label indicator matrix as a binary prediction.
precision = precision_score(y, predicted_cv, average='micro')  # 0.9662921348314607
recall = recall_score(y, predicted_cv, average='micro')  # 0.9662921348314607
f1 = f1_score(y, predicted_cv, average='micro')  # 0.9662921348314607


# Let's split into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.40, random_state=42)

# fit our model
lda_fit2 = lda.fit(X_train, y_train)
# predict
predicted_cv2 = cross_val_predict(lda_fit2, X_test, y_test, cv=10)

# confusion matrix
con_matrix2 = confusion_matrix(y_test, predicted_cv2)
# print(con_matrix2) # [[26  0  0][ 2 24  1] [ 0  0 19]


# Precision, Recall and F metric
precision2 = precision_score(y_test, predicted_cv2, average='micro')  # 0.9583333333333334
recall2 = recall_score(y_test, predicted_cv2, average='micro')  # 0.9583333333333334
f1_2 = f1_score(y_test, predicted_cv2, average='micro')  # 0.9583333333333334


# Plotting multi-dimensional data.
# LDA can be used for dimensionality reduction and that's a good approach to visualize the
# spread of classes in our data
lda_reduce = LinearDiscriminantAnalysis(n_components=2)
lda_transformed = pd.DataFrame(lda_reduce.fit_transform(X, y))

# graph our 3 class
plt.scatter(lda_transformed[y == 0][0], lda_transformed[y == 0][1], label='class 1', c='red')
plt.scatter(lda_transformed[y == 1][0], lda_transformed[y == 1][1], label='class 2', c='blue')
plt.scatter(lda_transformed[y == 2][0], lda_transformed[y == 2][1], label='class 3', c='yellow')

plt.legend(loc=3)
plt.show()