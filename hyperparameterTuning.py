from sklearn import metrics as metrics
import itertools

def tune(X_train, X_test, y_train, y_test, model, **kwargs):
    # Contains all the hyperparams to be tuned
    keys = []
    # Contains all the possible values of hyperparams
    lists = []

    for key, value in kwargs.items():
        lists.append(value)
        keys.append(key)

    # Maximum accuracy obtained so far
    max_accuracy = -1

    # Best set of params which result in maximum accuracy
    best_params = {}

    # For all combinations of hyperparams
    for el in itertools.product(*lists):

        # Create current set of test params
        test = {}
        for i in range(len(el)):
            test[keys[i]] = el[i]

        # Train the model on Z
        m = model(**test)
        m.fit(X_train, y_train)

        # Evaluate the model on Z'_1
        y_pred = m.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)

        # Update max_accuracy and best_params
        if (acc > max_accuracy):
            max_accuracy = acc
            best_params = test

    # Return the best set of params
    return best_params
