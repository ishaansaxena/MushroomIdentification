import numpy as np
import numpy.linalg as la
# Input: matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# vector y of scalar values, with n rows (samples), 1 column
# y[i] is the scalar value of the i-th sample
# Output: vector theta, with d rows, 1 column
# Example on how to call the function:
# import linreg
# theta = linreg.run(X,y)

def run(X,y):
	#print la.inv(X.T.dot(X)).dot(X.T).dot(y)
	return np.dot(la.pinv(X),y)

# X, y = createlinregdata.run(10,3)

# theta = run(X,y)
# print theta
