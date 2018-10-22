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
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC

    models = []
    models.append(("SVM-sigmoid", SVC, {'kernel'='sigmoid'}))
    models.append(("SVM-linear", SVC, {'kernel'='linear'}))
    models.append(("SVM-poly", SVC, {'kernel':'poly'}))
    models.append(("SVM-rbf", SVC, {'kernel':'rbf'}))

    # For each algorithm
    for name, model, kwargs in models:

        print "Model: %s | Time: " % (name),
        start = time.time()

        # Folds
        xrange = [2, 5]

        # Mean accuracy and std dev
        m = []
        s = []

        # For each fold size
        for i in xrange:
            Z = kfoldcv.run(X, y, model, i, **kwargs)
            m.append(np.mean(Z))
            s.append(np.std(Z))

        # Plot accuracy vs. fold size
        df = pd.DataFrame({
            'mean_accuracy': m,
            'std_dev': s
        }, index = xrange)

        lines = df.plot.line()
        filename = project.results + name + "_accuracy_vs_folds.png"
        plt.savefig(filename, bbox_inches='tight')

        print "%.3fs" % (time.time() - start)
