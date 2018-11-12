import matplotlib.pyplot as plt

import time
import numpy as np
import pandas as pd

import data
import project
import kfoldcv

if __name__ == '__main__':

    # Load (X, y) dataset
    X, y = data.load()

    # Get X as a numpy matrix
    X = X.values

    # Import models
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import Perceptron
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression

    models = []

    models.append(("Perceptron-alpha 0.0001", Perceptron, {'alpha':0.0001}))
    models.append(("Perceptron-alpha 1", Perceptron, {'alpha':1}))
    models.append(("Perceptron-alpha 100", Perceptron, {'alpha':100}))
    models.append(("Perceptron-pentalty l2", Perceptron, {'pentalty':'l2'}))
    models.append(("Perceptron-pentalty l1", Perceptron, {'pentalty':'l1'}))
    models.append(("Perceptron-pentalty elasticnet", Perceptron, {'pentalty':'elasticnet'}))

    models.append(("Adaboost-n_estimators 50", AdaBoostClassifier, {'n_estimators':50}))
    models.append(("Adaboost-n_estimators 100", AdaBoostClassifier, {'n_estimators':100}))
    models.append(("Adaboost-learning_rate 1", AdaBoostClassifier, {'learning_rate':1}))
    models.append(("Adaboost-learning_rate 10", AdaBoostClassifier, {'learning_rate':10}))
    models.append(("Adaboost-algorithm SAMME", AdaBoostClassifier, {'algorithm':'SAMME'}))
    models.append(("Adaboost-algorithm SAMME.R", AdaBoostClassifier, {'algorithm':'SAMME.R'}))

    models.append(("SVM-linear", SVC, {'kernel':'linear'}))
    models.append(("SVM-poly", SVC, {'kernel':'poly'}))
    models.append(("SVM-rbf", SVC, {'kernel':'rbf'}))

    models.append("LR", LogisticRegression, {})

    models.append(("XGB", XGBClassifier, {}))

    # For each algorithm
    for name, model, kwargs in models:

        print ("Model:\t%s" % (name))
        start = time.time()

        # Folds
        xrange = [2, 5, 10]

        # Mean accuracy and std dev
        m = []
        s = []

        # For each fold size
        for i in xrange:
            Z = kfoldcv.run(X, y, model, i, **kwargs)
            m.append(np.mean(Z))
            s.append(np.std(Z))
            print ("\t%d-fold: (mu=%.3f, sigma=%.3f)" % (i, np.mean(Z), np.std(Z)))

        # Plot accuracy vs. fold size
        df = pd.DataFrame({
            'mean_accuracy': m,
            'std_dev': s
        }, index = xrange)

        lines = df.plot.line()
        filename = project.results + name + "_accuracy_vs_folds.png"
        plt.savefig(filename, bbox_inches='tight')

        print ("\tTime: %.3fs" % (time.time() - start))
