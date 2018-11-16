import numpy as np
import pandas as pd

import project

def load(return_full_df=False):
    df = pd.read_csv(project.config['filename'])

    # Encode Ordinal Variables
    ordinal_columns = ['gill-spacing', 'gill-size', 'stalk-shape', 'ring-number', 'population', 'class']
    columns = ordinal_columns[:]

    for column in columns:
            df[column] = df[column].astype('category')

    columns = df.select_dtypes(['category']).columns
    df[columns] = df[columns].apply(lambda x: x.cat.codes)

    # Encoding Nominal Variables
    columns = ordinal_columns[:]

    for column in df:
        if column not in columns:
            dummies = pd.get_dummies(df.pop(column))
            column_names = [column + "_" + x for x in dummies.columns]
            dummies.columns = column_names
            df = df.join(dummies)

    if return_full_df:
        return df

    # Separate Data into X, y
    label = 'class'
    X = df.loc[:, df.columns != label]
    y = df[label].ravel()

    return X, y
