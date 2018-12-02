from sklearn.model_selection import train_test_split as split
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import time

import data
import models
import hyperparameterTuning
import kfoldcv
import hypothesisTest

if __name__ == '__main__':

    # Load dataset with reduced number of features. Note that Z = (X_train, y_train) and
    # Z_1 {union} Z_2 = (X_test, y_test) as described in Section 2 of the report.
    print("Preprocessing and loading dataset...")
    X_train, X_test, y_train, y_test = data.get_reduced_data(test_size=0.6, F=50)

    # Create two testing datasets Z_1, Z_2 from Z_ = (X_, y_)
    X_test_1, X_test_2, y_test_1, y_test_2 = split(
        X_test, y_test, test_size=0.66
    )

    # Get list m which contains all models and possible hyperparameters for each model.
    m = models.get_model_list()

    # Use model results to store results for each model
    model_results = []

    # For each model in the list of models
    for name, model, hyperparams in m:

        print("\nModel:\t%s" % (name))
        start = time.time()

        # Find optimal hyperparameters for current model by training on Z = (X_train, y_train)
        # and evaluation on Z_2 = (X_test_2, y_test_2)
        print("Finding optimal hyperparameters:")
        best_hyperparams = hyperparameterTuning.tune(X_train, X_test_1, y_train, y_test_1, model, **hyperparams)
        print(best_hyperparams)

        # Conduct k-Fold CV on the best hyperparams of the current model on Z_2
        print("\nRunning k-fold cross-validation on best hyperparameters:")

        # Fold sizes
        folds = [2, 5, 10]

        # Mean accuracy and std dev
        means = []
        stddevs = []

        # For each fold size
        for fold_size in folds:
            Z = kfoldcv.run(X_test_2, y_test_2, model, fold_size, **best_hyperparams)
            means.append(np.mean(Z))
            stddevs.append(np.std(Z))
            print("\t%d-fold: (mu=%.3f, sigma=%.3f)" % (fold_size, np.mean(Z), np.std(Z)))

            # Use 5 fold to compare models
            if fold_size == 5:
                model_results.append((name, np.mean(Z), np.std(Z)))

        # Plot accuracy vs. fold size
        df = pd.DataFrame({
            'mean_accuracy': means,
            'std_dev': stddevs
        }, index = folds)

        lines = df.plot.line()
        filename = "results/" + "model_accuracy_vs_folds_" + name + ".png"
        plt.title(name + str(best_hyperparams))
        plt.ylim(0,1.01)
        plt.savefig(filename, bbox_inches='tight')

        print("Time: %.3fs\n" % (time.time() - start))

    # Compare the best two models
    model_results.sort(key=lambda x: -x[1])

    for model_result in model_results:
        print(model_result)

    # Hypothesis Testing
    model1, mu1, sigma1 = model_results[0]

    # Testing results of 5 folds with 99% confidence
    n = 5
    alpha = 1 - 0.99
    print ""
    for i in range(1, len(model_results)):
        model2, mu2, sigma2 = model_results[i]
        print "%s > %s:" % (model1, model2), hypothesisTest.t_test(1-mu1, sigma1, 1-mu2, sigma2, n, alpha)
