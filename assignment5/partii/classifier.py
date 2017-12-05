################################################
###              classifier.py               ###
################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


###############  function  #################
def report_model_accuracy(model, X_train, y_train, X_test, y_test, other_acc=False):
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_pred=y_pred, y_true=y_test)
    if not other_acc:
        return(score)
    else:
        precision = precision_score(y_pred=y_pred, y_true=y_test)
        recall = recall_score(y_pred=y_pred, y_true=y_test)
        f1 = f1_score(y_pred=y_pred, y_true=y_test)
        return(score, precision, recall, f1)

def ten_fold_crossvalidation(model, X, y, other_acc=False):
    accuracies = list()
    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        # print(X_train, y_train)
        score = report_model_accuracy(model, X_train, y_train, X_test, y_test, other_acc)
        accuracies.append(score)
    return(accuracies)

def ave_report(result):
    accuracy = np.mean([item[0] for item in result])
    precision = np.mean([item[1] for item in result])
    recall = np.mean([item[2] for item in result])
    f1 = np.mean([item[3] for item in result])
    return(round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3))

###############  read data  ################
bc_data = pd.read_csv("/Users/minzhe/Documents/Note/6375MachineLearning/assignment5/partii/data/BreastCancerWisconsinDiagnostic.csv")
### check for any missing value
na_idx = list()
for idx, row in bc_data.iterrows():
    if any(row.apply(lambda x: x == 0)):
        na_idx.append(idx)
bc_data.drop(na_idx, inplace=True)
### scale
le = LabelEncoder()
y = le.fit_transform(bc_data.diagnosis)   # label
X = bc_data.iloc[:,2:]  # value
X = StandardScaler().fit_transform(X)

############  tune model parameters  ############
### deicision tree
depths = [4, 8, 12, 16, None]
ave_acc_tree = list()
for depth in depths:
    clf_tree = DecisionTreeClassifier(max_depth=depth)
    accuracies = ten_fold_crossvalidation(model=clf_tree, X=X, y=y)
    ave_acc_tree.append(np.mean(accuracies))

### neural network
layers = [(3,), (6,), (10,), (15,), (20,), (50,), (100,)]
ave_acc_nn = list()
for layer in layers:
    clf_nn = MLPClassifier(hidden_layer_sizes=layer)
    accuracies = ten_fold_crossvalidation(clf_nn, X=X, y=y)
    ave_acc_nn.append(np.mean(accuracies))

### svm
Cs = [0.05, 0.2, 1, 5, 20, 200]
ave_acc_svm = list()
for C in Cs:
    clf_svm = SVC(C=C)
    accuracies = ten_fold_crossvalidation(model=clf_svm, X=X, y=y)
    ave_acc_svm.append(np.mean(accuracies))

### logstic regression
Cs = [0.001, 0.01, 0.05, 0.2, 1, 5, 20, 200]
ave_acc_logit = list()
for C in Cs:
    clf_logit = LogisticRegression(C=C)
    accuracies = ten_fold_crossvalidation(clf_logit, X, y)
    ave_acc_logit.append(np.mean(accuracies))

### k nearest neighbors
ks = [2, 3, 4, 5, 6]
ave_acc_knn = list()
for k in ks:
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    accuracies = ten_fold_crossvalidation(clf_knn, X, y)
    ave_acc_knn.append(np.mean(accuracies))

### bagging
n_estimators = [4, 6, 8, 10, 15]
ave_acc_bagging = list()
for n in n_estimators:
    clf_bagging = BaggingClassifier(n_estimators=n)
    accuracies = ten_fold_crossvalidation(clf_bagging, X, y)
    ave_acc_bagging.append(np.mean(accuracies))

### random forest
n_estimators = [4, 6, 8, 10, 15]
ave_acc_rf = list()
for n in n_estimators:
    clf_rf = RandomForestClassifier(n_estimators=n)
    accuracies = ten_fold_crossvalidation(clf_rf, X, y)
    ave_acc_rf.append(np.mean(accuracies))

### adaboost
n_estimators = [10, 20, 30, 40, 50, 60]
ave_acc_adaboost = list()
for n in n_estimators:
    clf_adaboost = AdaBoostClassifier(n_estimators=n)
    accuracies = ten_fold_crossvalidation(clf_adaboost, X, y)
    ave_acc_adaboost.append(np.mean(accuracies))

#############  evaulate performance  ##############
### tree
clf_tree = DecisionTreeClassifier(max_depth=8)
result = ten_fold_crossvalidation(clf_tree, X, y, True)
tree_score = ave_report(result)

### neural network
clf_nn = MLPClassifier(hidden_layer_sizes=(20,))
result = ten_fold_crossvalidation(clf_nn, X, y, True)
nn_score = ave_report(result)

### SVM
clf_svm = SVC(C=5)
result = ten_fold_crossvalidation(clf_svm, X, y, True)
svm_score = ave_report(result)

### logistic regression
clf_logit = LogisticRegression(C=0.2)
result = ten_fold_crossvalidation(clf_logit, X, y, True)
logit_score = ave_report(result)

### KNN
clf_knn = KNeighborsClassifier(n_neighbors=4)
result = ten_fold_crossvalidation(clf_knn, X, y, True)
knn_score = ave_report(result)

### Bagging
clf_bagging = BaggingClassifier(n_estimators=10)
result = ten_fold_crossvalidation(clf_bagging, X, y, True)
bagging_score = ave_report(result)

### Random Forest
clf_rf = RandomForestClassifier(n_estimators=10)
result = ten_fold_crossvalidation(clf_rf, X, y, True)
rf_score = ave_report(result)

### AdaBoost
clf_adaboost = AdaBoostClassifier(n_estimators=50)
result = ten_fold_crossvalidation(clf_adaboost, X, y, True)
adaboost_score = ave_report(result)
