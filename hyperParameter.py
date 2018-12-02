import itertools
from sklearn import metrics as metrics

def tune(X_train, X_test, y_train, y_test, model, **kwargs):
	lists = []
	keys = []

	for key, value in kwargs.items():
		lists.append(value)
		keys.append(key)

	maxi = -1
	bestel = {}

	#Get all combinations
	for el in itertools.product(*lists):
		test = {}
		for i in range(len(el)):
			test[keys[i]] = el[i]

		#Train all combinations
		m = model(**test)
		m.fit(X_train, y_train)
		y_pred = m.predict(X_test)
		acc = metrics.accuracy_score(y_test, y_pred)
			
		#Update Best accuracy
		if (acc > maxi):
			maxi = acc
			bestel = test


	return bestel 


    
