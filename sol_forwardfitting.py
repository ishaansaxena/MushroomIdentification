import linreg as lr
import sys
import numpy as np

# Input: number of features F
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of scalar values, with n rows (samples), 1 column
# y[i] is the scalar value of the i-th sample
# Output: numpy vector of selected features S, with F rows, 1 column
# numpy vector thetaS, with F rows, 1 column
# thetaS[0] corresponds to the weight of feature S[0]
# thetaS[1] corresponds to the weight of feature S[1]
# and so on and so forth
def run(F,X,y):
    S = []
    thetaS = []
    z = np.zeros((len(X),1))
    for f in range(F):
        mins = float("inf")
        minsum = float("inf")
        thetagucci = 0
        for t in range(len(X)):
            inside = [i for i in xrange(len(X[0])) if i in S]
            data = X[:, inside]     
            if (len(S) == 0):
                z[t] = y[t]
            else:
                z[t] = y[t] - np.vdot(np.asarray(thetaS).reshape(len(thetaS), 1), data[t].reshape(1,len(thetaS)))
                
        for j in range(len(X[0])):
            if j not in S:
                data = X[:, j]
                thetaj = lr.run(data.reshape(len(X),1), y)

                sums = 0
                for t in range(len(X)):
                    sums += (z[t] - np.vdot(thetaj, data[t]))**2
                sums = 0.5*sums
                if (sums < minsum):
                    minsum = sums
                    mins = j
                    thetagucci = thetaj
        jhat = mins
        S.append(jhat)
        thetaS.append(thetagucci)

    return (np.asarray(S).reshape(F,1), np.asarray(thetaS).reshape(F,1))
    

