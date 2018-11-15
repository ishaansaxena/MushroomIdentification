import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import project
import numpy as np
# import featureSelection as fs

def FeatPlots():
	#Use Old Data for labels
	df = pd.read_csv(project.config['filename'])
	
	cols = df.select_dtypes(include=['object']).columns

	# Get Plots for Feature Frequencies
		
	for c in cols:
		labs = df[c].value_counts().axes[0].tolist()
		ind = np.arange(len(labs))
		s = []
		t = []
		
		sns.countplot(x = c, data = df)
		plt.title('Feature Frequencies')
		plt.show()

		for l in labs:
			numpois = 0
			tot = len(df[df[c] == l].index)
			for i in range(len(df[c])):
				if ((df[c][i] == l) & (df['class'][i] == 'p')):
					numpois += 1
			s.append(numpois)
			t.append(tot-numpois)

		width = 0.40
		fig, ax = plt.subplots(figsize=(12,7))
	
		edible_bars = ax.bar(ind, t , width, color='g')
		poison_bars = ax.bar(ind+width, s , width, color='r')

		#Add some text for labels, title and axes ticks
		ax.set_xticks(ind + width / 2) #Positioning on the x axis
		ax.set_xticklabels(labs, fontsize = 12)
		ax.legend((edible_bars,poison_bars),('edible','poisonous'),fontsize=17)
		plt.title('Edible v/s Poisonous for %s' %c)
		plt.show()
		
FeatPlots()
