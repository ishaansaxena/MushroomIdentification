from sklearn.metrics import accuracy_score

import numpy as np

def run(X, y, model, k, verbose=False, *args, **kwargs):
    # Get n (samples) and d (dimensions)
    (n, d) = X.shape

    # Initialize Z
    Z = np.zeros((k, 1))

    if verbose:
        print "%d-Fold CV\tModel: %s" % (k, model)

    # For each fold
    for i in range(k):
        # Find sets S and T
        lower = np.floor(float(n) * i / k).astype('int64')
        upper = np.floor(float(n) * (i + 1) / k - 1).astype('int64')
        T = range(lower, upper + 1)
        S = range(0, lower) + range(upper + 1, n)

        # Get training data
        X_train = X[S, :]
        y_train = y[S]

        # Fit model
        m = model(**kwargs)
        m.fit(X_train, y_train)

        # Check prediction accuracy
        y_pred = m.predict(X[T])

        # Update Z values based on accuracy score
        Z[i] = accuracy_score(y[T], y_pred)

        if verbose:
            print "\t\tIteration (%d/%d)\tAccuracy Score: %f" % (i + 1, k, Z[i])

    # Return k-Fold results
    return Z
