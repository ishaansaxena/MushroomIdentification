import pandas as pd
import matplotlib.pyplot as plt
import project
import data
import numpy as np
import seaborn as sns

#def chi_sq_test(df, col1, col2):
#	result_list[]
#	for car in range(int(df[col[1].min()), int(df[var2].max())+1):
#		result_list.append(create_list(df, col1, cat, col2))
#	return scs.chi2_contigency

def sorting(numbers):
  return sorted(numbers[1],key = abs)

def correlationPlots():
   #Use old data for labels
   #df = pd.read_csv(project.config['filename'])
   
   Xy = data.load(return_full_df = True) 
   
   corr = Xy.corr()

   c = Xy.corr().abs()
 
   s = c.unstack()
   so = s.sort_values(kind="quicksort",ascending = False)
  
   
   so = so.dropna()
   #so = so[so.col1 != so.col2]

   # soDescending = so[110:130]
   
   
   # print(soDescending) 
   #print(so.index)
   
   so = c.stack().reset_index()
   so['A_vs_B'] = so.level_0 + '_' + so.level_1
   so.columns = ['A','B','correlation', 'A_vs_B'] 
   #so = ['A','B','correlation', 'A_vs_B']
   #so = so.loc[so.A != so.B, ['A_vs_B','correlation']]

   #classFeatures = so.loc[so.A == 'class' | so.B == 'class', ['A_vs_B','correlation']]
   
   
   sns.heatmap(Xy.corr())
   filename = project.results + "heatmap"
   plt.savefig(filename,bbox_inches = 'tight')
   filename = project.results + "corrMatrix"
   so.to_csv(filename)

   #Class vs feature correlation

   df = data.load(True);
   ClassFeatures = []
   ClassCorr = []
   
   for column in df.columns:
      corr = df[column].corr(df['class'])
      ClassCorr.append(corr)
      ClassFeatures.append(column)
   
   dataS = pd.DataFrame({
	'feature' : ClassFeatures,
	'corr' : ClassCorr,
	'abscorr' : np.abs(ClassCorr)
   })

   result = dataS.groupby(["feature"])['abscorr'].aggregate(np.median).reset_index().sort_values('abscorr')
   #print(dataS, order = result['feature'])
   dataS = dataS.sort_values(['abscorr'], ascending=False)
   print dataS   

   #plt.figure()
   #sns.barplot(x = 'feature', y = 'corr', data = dataS, order = result['feature'])   
   filename = project.results + "class vs features"
   #plt.show()
   #ClassFeatures = np.array(ClassFeatures)



correlationPlots()
