from sklearn.model_selection import train_test_split

import data
import greedySubset as gs
import project as pr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def run(F):
    # Load encoded data
    X, y = data.load()

    # Split data into Test-Train Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    # Get subset of features
    S, theta = gs.run(F, X_train.values, y_train)

    # Print S, theta
    print(S)
    print(theta)

    # Create a dataframe of features and weights
    df = pd.DataFrame({
        'feature' : S,
        'weight' : theta,
        'weightabs': np.abs(theta)
        })

    # Plot Greedy Subset
    plt.figure()

    # Filter only those features with weights more than 1/2 the mean weight
    m = (np.mean(df['weightabs'].values)/2)
    df = df[df['weightabs'] > m]

    # Sort features by weights
    result = df.groupby(["feature"])['weightabs'].aggregate(np.median).reset_index().sort_values('weightabs')

    # Plot features and weights
    sns.barplot(x='feature', y="weight", data=df, order=result['feature'])

    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.title('Greedy Subset')
    plt.show()

run(107)
