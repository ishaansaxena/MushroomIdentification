import matplotlib.pyplot as plt

import time
import numpy as np
import pandas as pd

import data
import project
import kfoldcv
import metrics

if __name__ == '__main__':

    # Load (X, y) dataset
    X, y = data.load()

    # Get X as a numpy matrix
    X = X.values

    # Import models
    # from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import Perceptron
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression

    models = []

    models.append(("Perceptron", Perceptron, {'max_iter':500}))
    models.append(("Adaboost-n_estimators 10", AdaBoostClassifier, {'n_estimators':10}))
    models.append(("Adaboost-n_estimators 50", AdaBoostClassifier, {'n_estimators':50}))
    models.append(("SVM-linear", SVC, {'kernel':'linear'}))
    models.append(("SVM-poly", SVC, {'kernel':'poly'}))
    models.append(("SVM-rbf", SVC, {'kernel':'rbf'}))
    models.append(("LR", LogisticRegression, {}))

    # For each algorithm
    for name, model, kwargs in models:

        print ("Model:\t%s" % (name))
        start = time.time()

        # Mean accuracy and std dev
        m = []
        s = []

        # For each fold size

        Z = kfoldcv.run(X, y, model, 5, **kwargs)
        m.append(np.mean(Z))
        s.append(np.std(Z))
        print ("\t%d-fold: (mu=%.3f, sigma=%.3f)" % (5, np.mean(Z), np.std(Z)))


        print ("\tTime: %.3fs" % (time.time() - start))
