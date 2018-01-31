from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np

# let's try to identify Iris-Virginica(targe = 2) just by using petal width feature.
# This is a basic example of single class classification

iris_np = load_iris()
X = iris_np['data'][:, 3:]  # peta; width
y = (iris_np['target'] == 2).astype(np.int)  # make the output = 1 if Iris-Virginica, else 0

# fit out model
logit = LogisticRegression()
logit.fit(X, y)

# find the decision boundary of each class(where the two lines meet)
X_test = np.linspace(0, 3, 1000).reshape(-1, 1)  # make synthetic test data. 0<=petal widht<=3
y_prob = logit.predict_proba(X_test)  # get the probabilities of each class
# plt.plot(X_test, y_prob[:, 1], 'g-', label='Iris-Virginica')
# plt.plot(X_test, y_prob[:, 0], 'b--', label='Not Iris-Virginica')
# plt.show() # uncomment to show graph


# Model accuracy

# 1) confusion matrix --> perfect predictor: all but the main diagonal are zeros
predicted = cross_val_predict(logit, X, y, cv=10)
matrix = confusion_matrix(y, predicted)  # [[95  5][ 3 47]]
# print(matrix)


# 2) Precision VS Recall
precision = precision_score(y, predicted)  # 0.9038461538461539
recall = recall_score(y, predicted)  # 0.94

# 3) F1 score or harmonic mean of precision and recall
# Combines precision and recall into a single formula. Useful when you care about
# the overall performance and not so much about them individually
f1 = f1_score(y, predicted)  # 0.9215686274509804


# 4) ROC curve ==> True positive rate VS False positive rate
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


y_scores = cross_val_predict(logit, X, y, cv=10, method='decision_function')
fpr, tpr, thresholds = roc_curve(y, y_scores) # gets the True positive and false positive rate.
# plot_roc_curve(fpr, tpr) # matplot code to graph
# plt.show()

# how to compare two classifiers, use the area under the curve. area = { 1 ==> perfect, 0.5 ==> random}
area_under_roc = roc_auc_score(y, y_scores)
