import matplotlib.pyplot as plt

import time
import numpy as np
import pandas as pd
import scikitplot as skplt

import data
import project
import kfoldcv
import metrics

if __name__ == '__main__':

    # Load (X, y) dataset
    X, y = data.load()

    # Get X as a numpy matrix
    # X = X.values

    # Import models
    # from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import Perceptron
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_curve, auc


    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=4)
    model_LR= SVC(kernel='poly', probability=True)
    model_LR.fit(X_train,y_train)
    model_LR.multi_class='ovr'
    model_LR.solver='liblinear'
    model_LR.n_jobs=1

    y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
    y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
    print(model_LR.score(X_test, y_pred))
    confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
    print(confusion_matrix)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
