# Input: number of features F
# matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# vector y of scalar values, with n rows (samples), 1 column
# y[i] is the scalar value of the i-th sample
# Output: vector of selected features S, with F rows, 1 column
# vector thetaS, with F rows, 1 column
# thetaS[0] corresponds to the weight of feature S[0]
# thetaS[1] corresponds to the weight of feature S[1]
# and so on and so forth
import numpy as np
import linreg

def run(F,X,y):
    # Your code goes here
    S = []
    theta_s = np.empty((0,0))
    # print theta_s
    J = [i for i in range(0,len(X[0]))] # no of features
    for f in range(1,F+1):
        # calculate z
        X_s = X[:,S]
        if len(S) == 0:
            z = np.copy(y)
        else:
            z = y - np.dot(X_s,theta_s)

        # get list of features to iterate on.
        f_iter = []
        for t in J:
            if not (t in S):
                f_iter.append(t)
        
        best_j = None
        max_effect = float("-inf")
        best_theta_j = None
        for j in f_iter:
            # get X_ = x_{t,j}
            X_j_temp = X[:,j]
            X_j = X_j_temp.reshape(len(X_j_temp),1)
            
            effect = (-1) * np.asscalar(np.dot(z.T,X_j))
            effect = abs(effect)
            if (effect > max_effect):
                max_effect = effect
                best_j = j
                best_X_j = X_j

        best_X_j_temp = X[:,best_j]
        best_X_j = best_X_j_temp.reshape(len(best_X_j_temp),1)
        best_theta_j = linreg.run(best_X_j,z)
        S.append(best_j)
        # update thetaS with concatenation of best_theta_j
        theta_s = np.append(theta_s,best_theta_j)
        theta_s = theta_s.reshape(len(theta_s),1)

    return (np.asarray(S).reshape(F,1), theta_s)