import data
import greedySubset as gr
import project as pr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def run(F):
    c_X, c_Y = data.load()
    print(c_Y)

    X = c_X.values

    #Greedy Subset
    t2, t2_S = gr.run(F,X,c_Y)
    
    print(t2)
    print(t2_S)
    
    df = pd.DataFrame({
        'feature' : t2,
        'weight' : t2_S,
        'weightabs': np.abs(t2_S)
        })

    #Plot Greedy Subset
    plt.figure()

    m = (np.mean(df['weightabs'].values)/2)
    df = df[df['weightabs'] > m]
    
    result = df.groupby(["feature"])['weightabs'].aggregate(np.median).reset_index().sort_values('weightabs')

    sns.barplot(x='feature', y="weight", data=df, order=result['feature'])

    # df.groupby(['feature']).median().sort_values("weightabs").plot.bar()
    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.title('Greedy Subset')
    plt.show()


run(107)
