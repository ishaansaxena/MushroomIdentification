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
    from sklearn.ensemble import AdaBoostClassifier

    models = []
    models.append(("Adaboost-n_estimators 50", AdaBoostClassifier, {'n_estimators':50}))
    models.append(("Adaboost-n_estimators 100", AdaBoostClassifier, {'n_estimators':100}))
    models.append(("Adaboost-learning_rate 1", AdaBoostClassifier, {'learning_rate':1}))
    models.append(("Adaboost-learning_rate 10", AdaBoostClassifier, {'learning_rate':10}))
    models.append(("Adaboost-algorithm SAMME", AdaBoostClassifier, {'algorithm':'SAMME'}))
    models.append(("Adaboost-algorithm SAMME.R", AdaBoostClassifier, {'algorithm':'SAMME.R'}))

    # For each algorithm
    for name, model, n_estimators in models:

        print "Model:\t%s" % (name)
        start = time.time()

        # Folds
        xrange = [2, 5, 10, 20]

        # Mean accuracy and std dev
        m = []
        s = []

        # For each fold size
        for i in xrange:
            Z = kfoldcv.run(X, y, model, i)
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
