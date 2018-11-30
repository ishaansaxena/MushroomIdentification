import numpy as np
import hypothesisTest as hypt

import data
import kfoldcv

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

    model_results = []

    # For each algorithm
    for name, model, kwargs in models:

        Z = kfoldcv.run(X, y, model, 5, **kwargs)
        model_results.append((name, np.mean(Z), np.std(Z)))


    k = 5
    alpha = 0.05

    for m1, mean_1, std_dev_1 in model_results :
        for m2, mean_2, std_dev_2 in model_results :
            if m1 == m2:
                continue
            if hypt.hypothesisTesting(mean_1, std_dev_1, mean_2, std_dev_2, k, alpha):
                print m1 + ' over ' + m2
