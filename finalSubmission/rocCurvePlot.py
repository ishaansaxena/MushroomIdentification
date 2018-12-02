import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

import time
import numpy as np
import pandas as pd

import data
import project
import kfoldcv

def run(X, X_, y, y_, model, plotTitle, *args, **kwargs):
    m = model(**kwargs)

    m.fit(X, y)
    m.probability = True

    # This will give you positive class prediction probabilities
    y_prob = m.predict_proba(X_)[:,1]

    # This will threshold the probabilities to give class predictions.
    y_pred = np.where(y_prob > 0.5, 1, 0)
    print(m.score(X_, y_pred))

    # Get confusion matrix
    confusion_matrix=metrics.confusion_matrix(y_, y_pred)
    print(confusion_matrix)

    # Get ROC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)

    # Plot figure
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color='red',label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(plotTitle)
    plt.xlim(1, 0)
    plt.ylim(0, 1.001)
    filename = "results/" + "roc_" + plotTitle + ".png"
    plt.savefig(filename)

if __name__ == '__main__':

    import data
    X, X_, y, y_ = data.get_reduced_data(0.8, 50)

    # Adaboost
    from sklearn.ensemble import AdaBoostClassifier
    hyperparams = {'n_estimators': 20, 'learning_rate': 0.5, 'algorithm': 'SAMME.R'}
    run(X, X_, y, y_, AdaBoostClassifier, "Adaboost", **hyperparams)

    # LogisticRegression
    from sklearn.linear_model import LogisticRegression
    hyperparams = {'penalty': 'l1', 'C': 1}
    run(X, X_, y, y_, LogisticRegression, "Logistic Regression", **hyperparams)

    # SVM Linear
    from sklearn.svm import SVC
    hyperparams = {'kernel': 'linear', 'C': 1, 'probability': True}
    run(X, X_, y, y_, SVC, "SVM-Linear", **hyperparams)

    # SVM Polynomial
    hyperparams = {'kernel': 'poly', 'C': 1, 'degree': 2,'probability': True}
    run(X, X_, y, y_, SVC, "SVM-Poly", **hyperparams)

    # SVM RBF
    hyperparams = {'kernel': 'rbf', 'C': 1, 'probability': True}
    run(X, X_, y, y_, SVC, "SVM-RBF", **hyperparams)
