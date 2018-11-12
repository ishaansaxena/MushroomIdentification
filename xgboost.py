import xgboost as xgb
from xgboost import XGBClassifier

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
    # from sklearn.svm import SVC

    models = []
    # models.append(("SVM-linear", SVC, {'kernel':'linear'}))
    # models.append(("SVM-poly", SVC, {'kernel':'poly'}))
    # models.append(("SVM-rbf", SVC, {'kernel':'rbf'}))
    models.append(("XGB", XGBClassifier, {}))

    # For each algorithm
    for name, model, kwargs in models:

        print "Model:\t%s" % (name)
        start = time.time()

        # Folds
        xrange = [2, 5, 10, 20]

        # Mean accuracy and std dev
        m = []
        s = []

        # For each fold size
        for i in xrange:
            Z = kfoldcv.run(X, y, model, i, **kwargs)
            m.append(np.mean(Z))
            s.append(np.std(Z))
            print "\t%d-fold: (mu=%.3f, sigma=%.3f)" % (i, np.mean(Z), np.std(Z))

        # Plot accuracy vs. fold size
        df = pd.DataFrame({
            'mean_accuracy': m,
            'std_dev': s
        }, index = xrange)

        lines = df.plot.line()
        filename = project.results + name + "_accuracy_vs_folds.png"
        plt.savefig(filename, bbox_inches='tight')

        print "\tTime: %.3fs" % (time.time() - start)
