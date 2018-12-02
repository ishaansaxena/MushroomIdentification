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
    X, X_, y, y_ = data.get_reduced_data(test_size=0.6)

    # Import models
    # from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import Perceptron
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression

    models = []

    models.append(("Adaboost-n_estimators_10", AdaBoostClassifier, {'n_estimators':10}))
    models.append(("Adaboost-n_estimators_20", AdaBoostClassifier, {'n_estimators':20}))
    models.append(("SVM-linear", SVC, {'kernel':'linear'}))
    models.append(("SVM-poly", SVC, {'kernel':'poly'}))
    models.append(("SVM-rbf", SVC, {'kernel':'rbf'}))
    models.append(("LR", LogisticRegression, {}))

    # For each algorithm
    for name, model, kwargs in models:

        print ("Model:\t%s" % (name))
        start = time.time()

        # Folds
        folds = [2, 5, 10]

        # Mean accuracy and std dev
        m = []
        s = []

        # For each fold size
        for fold_size in folds:
            Z = kfoldcv.run(X, y, model, fold_size, **kwargs)
            m.append(np.mean(Z))
            s.append(np.std(Z))
            print ("\t%d-fold: (mu=%.3f, sigma=%.3f)" % (fold_size, np.mean(Z), np.std(Z)))

        # Plot accuracy vs. fold size
        df = pd.DataFrame({
            'mean_accuracy': m,
            'std_dev': s
        }, index = folds)

        lines = df.plot.line()
        filename = "results/" + "model_accuracy_vs_folds_" + name + ".png"
        plt.title(name)
        plt.ylim(0.8,1.0001)
        plt.savefig(filename, bbox_inches='tight')

        print ("\tTime: %.3fs" % (time.time() - start))
