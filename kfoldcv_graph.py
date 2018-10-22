import matplotlib.pyplot as plt

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
    models.append(("LR", LogisticRegression))
    models.append(("CART", DecisionTreeClassifier))
    models.append(("SVM", SVC))

    # For each algorithm
    for name, model in models:

        # Folds [2, 4, 6, 8, 10, 15, 20, 25, 50, 75, 100]
        xrange = range(2, 10, 2) + range(10, 25, 5) + range(25, 125, 25)

        # Mean accuracy and std dev
        m = []
        s = []

        # For each fold size
        for i in xrange:
            Z = kfoldcv.run(X, y, model, i)
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
