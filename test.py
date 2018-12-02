import hyperparameterTuning as hp
import data
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

def run():
    models = []
    X, y = data.load()
    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size = 0.2,
            random_state = 42
        )

    models.append(("SVM", SVC,{'C': [0.1, 1.0, 10], 'kernel':['linear','poly','rbf']}))
    models.append(("LR", LogisticRegression, {}))

    for name, model, kwargs in models:
            e = hp.tune(X_train, X_test, y_train, y_test, model, **kwargs)
            print e


run()
