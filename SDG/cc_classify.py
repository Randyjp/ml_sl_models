import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

RANDOM_STATE = 42  # the meaning of life


def peek_digit(digit):
    """Display an image from the MNIST DS"""
    digit_image = digit.reshape(28, 28)
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()


def load_split_data(testing_percent):
    # 70k handwritten digits by HS students and USCB employees
    mnist = fetch_mldata('MNIST ORIGINAL')

    # Images are 28x28. Thus, each image has 784 features(one for each pixel).
    X, y = mnist.data, mnist.target
    # print(X.shape) # (70000, 784)
    # print(y.shape) # (70000,)

    # split into training
    return train_test_split(X, y, test_size=testing_percent, random_state=RANDOM_STATE)


def cross_val_results_score(model, X, y):
    score = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print("The 10-k cross validation score are ==>", score)


def confusion_matrix_results(y_real, y_pred):
    matrix = confusion_matrix(y_real, y_pred)
    print('############ Confusion Matrix ###############')
    print(matrix)


def precision_recall_f1_results(y_real, y_pred):
    print('####################################')
    precisison = precision_score(y_real, y_pred)
    recall = recall_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)

    print("Precision score: ", precisison)
    print("Recall score: ", recall)
    print("F1 score: ", f1)


def roc_curve_results(model, X_test, y_test):
    print('############ ROC Curve ###############')
    y_scores = cross_val_predict(model, X_test, y_test, cv=10, method='decision_function')
    # false positive rate, true positive rate
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # plot
    plt.plot(fpr, tpr, linewidth=2)
    # roc curve is between 0-1
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.show()


def random(X_train, X_test, y_train, y_test):
    random_forest = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
    random_forest.fit(X_train, y_train)
    scores = cross_val_score(random_forest, X_test, y_test, cv=10, scoring='accuracy')
    print('############ Random Forest Scores for all classes using 10-k cv ###############')
    print(scores)

def main_process():
    X_train, X_test, y_train, y_test = load_split_data(testing_percent=.30)
    # peek_digit(X_train[8])  # take a look to one digit, could be more but one for now

    # We'll try to decide between two classes.
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    # fit our model
    sdg = SGDClassifier(random_state=RANDOM_STATE, max_iter=5)
    sdg.fit(X_train, y_train_5)
    # print(sdg.predict(X_test[3].reshape(1, -1)))

    # cross val predict
    predicted_cv = cross_val_predict(sdg, X_test, y_test_5, cv=10)

    # cross validation [only 10% of images are 5's so accuracy will be high(meh)]
    cross_val_results_score(sdg, X_train, y_train_5)
    # get the confusion matrix
    confusion_matrix_results(y_test_5, predicted_cv)
    # get more metrics
    precision_recall_f1_results(y_test_5, predicted_cv)
    # roc curve
    roc_curve_results(sdg, X_test, y_test_5)
    # random forest
    random(X_train, X_test, y_train, y_test)


# test code
main_process()
