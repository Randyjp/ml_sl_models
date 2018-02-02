# Quadratic Discriminant analysis

# When LDA|QDA perform the best?
#
# 1)	The classes are well separated
# 2)	If n (# of samples) is small and the distribution of the predictors in X is approximately normal.
# 3)	For more than two response classes since it provides low-dimensional views of the data.


from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

wine = load_wine()
X = wine.data
y = wine.target

# fit the model
qda = QuadraticDiscriminantAnalysis()
qda_fit = qda.fit(X, y)
# print(lda_fit)

# compute accuracy metrics, keep in mind this is ONLY the training error. We might be over|under fitting

# confusion matrix
predicted_cv = cross_val_predict(qda_fit, X, y, cv=10)
con_matrix = confusion_matrix(y, predicted_cv)
# print(con_matrix)  # [58,  1,  0], [ 1, 70,  0],[ 0,  2, 46]])

# Precision, Recall and F metric

# micro-averaging ==> considering each element of the label indicator matrix as a binary prediction.
precision = precision_score(y, predicted_cv, average='micro')  # 0.98
recall = recall_score(y, predicted_cv, average='micro')  # 0.98
f1 = f1_score(y, predicted_cv, average='micro')  # 0.98


# Let's split into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.40, random_state=42)

# fit our model
qda_fit2 = qda.fit(X_train, y_train)
# predict
predicted_cv2 = cross_val_predict(qda_fit2, X_test, y_test, cv=10)

# confusion matrix
con_matrix2 = confusion_matrix(y_test, predicted_cv2)
# print(con_matrix2) # [25,  1,  0], [ 1, 26,  0], [ 0,  6, 13]])


# Precision, Recall and F metric
precision2 = precision_score(y_test, predicted_cv2, average='micro')  # 0.89
recall2 = recall_score(y_test, predicted_cv2, average='micro')  # 0.89
f1_2 = f1_score(y_test, predicted_cv2, average='micro')  # 0.89
