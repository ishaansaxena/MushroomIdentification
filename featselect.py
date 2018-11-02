import data
import sol_greedysubset as gr
import sol_forwardfitting as ff
import sol_myopicfitting as mp
import project as pr
import matplotlib.pyplot as plt
import numpy as np

def run(F):
	c_X, c_Y = data.load(pr.config['filename'],pr.config['label'])
	
	X = c_X.values

	#ForwardFitting
	t1, t1_S = ff.run(F,X,c_Y)
	
	#Greedy Subset
	t2, t2_S = gr.run(F,X,c_Y)

	#MyopicFitting
	# t3, t3_S = mp.run(F,X,y)

	#Plot Greedy Subset
	plt.figure()
	plt.bar(range(t2_S.size), t2_S)
	plt.xlabel('Feature')
	plt.ylabel('Weight')
	plt.title('Greedy Subset')
	plt.show()

	#Plot Forward Fitting
	plt.figure()
	plt.bar(range(t1_S.size), t1_S)
	plt.xlabel('Feature')
	plt.ylabel('Weight')
	plt.title('Forward Fitting')
	plt.show()

	#Plot Myopic Subset
	# plt.figure()
	# plt.bar(range(t3_S.size), t3_S)
	# plt.xlabel('Feature')
	# plt.ylabel('Weight')
	# plt.title('Myopic Fitting')
	# plt.show()

run(22)