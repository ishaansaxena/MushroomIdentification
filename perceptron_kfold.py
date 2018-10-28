import matplotlib.pyplot as plt

import time
import numpy as np
import pandas as pd

import data
import project
import kfoldcv

if __name__ == '__main__':

    # Load (X, y) dataset
    X, y = data.load(project.config['filename'], project.config['label'])

    # Get X as a numpy matrix
    X = X.values

    # Import models
    from sklearn.linear_model import Perceptron

    models = []
    models.append(("Perceptron-alpha 0.0001", Perceptron, {'alpha': 0.0001}))
    models.append(("Perceptron-alpha 1", Perceptron, {'alpha': 1}))
    models.append(("Perceptron-alpha 100", Perceptron, {'alpha': 100}))
    models.append(("Perceptron-pentalty l2", Perceptron, {'pentalty': 'l2'}))
    models.append(("Perceptron-pentalty l1", Perceptron, {'pentalty': 'l1'}))
    models.append(("Perceptron-pentalty elasticnet", Perceptron, {'pentalty': 'elasticnet'}))

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
