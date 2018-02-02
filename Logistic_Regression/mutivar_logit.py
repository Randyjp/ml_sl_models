from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, train_test_split
# let's try to identify Iris-Virginica(targe = 2) just by using petal width feature.
# This is a basic example of single class classification

iris_np = load_iris()
X = iris_np.data
y = iris_np.target

# divide  dataset into test(35%) and train(65%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.35)

# fit out model but since we now have 4 predictors we need to use a generalized model of the logistic
# regression, called Softmax regression. Set the multi_class hyper-parameter to "multinomial". Moreover,
# we need a solver that supports softmax, like lbfgs
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
softmax.fit(X_train, y_train)

y_test_proba = softmax.predict_proba(X_test)
y_test_score = softmax.decision_function(X_test)
y_test_predict = softmax.predict(X_test)

# # Model accuracy
#
# # 1) confusion matrix --> perfect predictor: all but the main diagonal are zeros
predicted_cross_val = cross_val_predict(softmax, X, y, cv=10)
matrix = confusion_matrix(y_test, y_test_predict)  # [[19  0  0] [ 0 17  1]
# print(matrix)
#
#
# # 2) Precision VS Recall
# precision = precision_score(y_train, predicted_cross_val, average='micro')  # 0.9733333333333334
# recall = recall_score(y_train, predicted_cross_val, average='micro')  # 0.9733333333333334
precision = precision_score(y_test, y_test_predict, average='micro')  # 1.0
recall = recall_score(y_test, y_test_predict, average='micro')  # 1.0

#
# # 3) F1 score or harmonic mean of precision and recall
# # Combines precision and recall into a single formula. Useful when you care about
# # the overall performance and not so much about them individually
# f1 = f1_score(y, predicted_cross_val, average='micro')  # 0.9215686274509804
f1 = f1_score(y_test, y_test_predict, average='micro')  # 1.0
#
#

