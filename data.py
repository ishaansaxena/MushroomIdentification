import numpy as np
import pandas as pd

def load(filename, label):
    # Read CSV File
    df = pd.read_csv(filename)

    # Remove N/A Values
    df = df.dropna()

    # Change categorical columns to numeric values
    for column in df:
        df[column] = df[column].astype('category')

    columns = df.select_dtypes(['category']).columns
    df[columns] = df[columns].apply(lambda x: x.cat.codes)

    # Separate X, y from df
    X = df.loc[:, df.columns != label]
    y = df[label].ravel()

    return X, y
