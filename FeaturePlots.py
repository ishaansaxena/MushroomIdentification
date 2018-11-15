import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import project
import featureSelection as fs

def FeatPlots():
	#Use Old Data for labels
	df = pd.read_csv(project.config['filename'])
	
	cols = df.select_dtypes(include=['object']).columns

	#Get Plots for Feature Frequencies
	for c in cols:
		sns.countplot(x = c, data = df)
		plt.title('Feature Frequencies')
		plt.show()
  

FeatPlots()
