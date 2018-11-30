from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import project
import greedySubset as gs

def load(return_full_df=False):
    df = pd.read_csv(project.config['filename'])
    df = encode(df)

    if return_full_df:
        return df

    # Separate Data into X, y
    label = 'class'
    X = df.loc[:, df.columns != label]
    y = df[label].ravel()

    return X, y

def encode(df):
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

    return df

# Input:    F = Number of Features to be selected
#           test_size = size of test subset
# Output:   Z = training set with reduced features
#           Z_ = testing set with reduced features
#           or
#           X_train, X_test, y_train, y_test
def get_reduced_data(test_size, F, return_full_df=False):
    # Load encoded data
    X, y = load()

    # Split data into Test-Train Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Get subset of features
    S, theta = gs.run(F, X_train.values, y_train)

    # Choose subset of X_train and X_test
    X_train = X_train.iloc[:, S]
    X_test = X_test.iloc[:, S]

    if not return_full_df:
        return X_train, X_test, y_train, y_test

    # Merge to form df if full dfs are required
    label = 'class'
    X_train[label] = y_train
    X_test[label] = y_test
    return X_train, X_test
