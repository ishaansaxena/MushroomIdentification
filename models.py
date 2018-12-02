# Import models which will be used.
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

def get_model_list():
    # Add a tuple of (model_name, model, hyperparameters) to the list of models.
    models = []

    # AdaBoostClassifier
    models.append(("Adaboost", AdaBoostClassifier, {
        'n_estimators': [3, 5, 10, 15, 20],
        'learning_rate': [0.1, 0.5, 1],
        'algorithm': ['SAMME', 'SAMME.R']
    }))

    # SVM
    models.append(("SVM-linear", SVC, {
        'kernel': ['linear'],
        'C': [0.01, 0.05, 0.1, 1, 10],
    }))

    # models.append(("SVM-poly", SVC, {
    #     'kernel': ['poly'],
    #     'C': [0.01, 0.05, 0.1, 1],
    #     'degree': [2, 3, 4]
    # }))

    models.append(("SVM-rbf", SVC, {
        'kernel': ['rbf'],
        'C': [0.01, 0.05, 0.1, 1],
    }))

    # LogisticRegression
    models.append(("LR", LogisticRegression, {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.05, 0.1, 1],
        'solver': ['liblinear', 'saga']
    }))


    return models
