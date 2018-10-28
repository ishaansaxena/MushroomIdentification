import numpy as np

import linreg

# Input: number of features F
#        numpy matrix X of features, with n rows (samples), d columns (features)
#           X[i, j] is the j-th feature of the i-th sample
#        numpy vector y of scalar values, with n rows (samples), 1 column
#           y[i] is the scalar value of the i-th sample
# Output: numpy vector of selected features S, with F rows, 1 column
#         numpy vector thetaS, with F rows, 1 column
#           thetaS[0] corresponds to the weight of feature S[0]
#           thetaS[1] corresponds to the weight of feature S[1]
#           and so on and so forth
def run(F, X, y):
    # Get n, d from data
    (n, d) = X.shape

    # Initialize S as an empty set
    S = []
    thetaS = []

    # For f in [0, F - 1]:
    for f in range(F):
        # Get all the z values
        if len(S) == 0:
            z = y
        else:
            tS = np.array(thetaS).reshape((f, 1))
            z = y - X[:, S].dot(tS)

        # Initialize empty J
        J = []
        theta = []

        # For each j not in S
        for j in range(d):
            # If j is in the set already, skip j
            if j in S:
                J.append(float("inf"))
                theta.append(0)
                continue

            # Get theta value from linreg
            theta_j = linreg.run(X[:, j].reshape((n, 1)), z)
            theta.append(theta_j)

            # Get J values from all t
            J.append(np.sum([
                ((z[t] - theta_j.dot(X[t, j])) ** 2)/2.0
                for t in range(n)
            ]))

        # Get argmin for J
        j = np.argmin(J)

        # Add j to S
        S.append(j)

        # Add theta[j] to thetaS
        thetaS.append(theta[j])

    # Reshape to match criteria
    S = np.array(S).reshape((F, 1))
    thetaS = np.array(thetaS).reshape((F, 1))

    return (S, thetaS)
