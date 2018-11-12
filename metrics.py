import numpy as np

def accuracy_score(y, yt):
    return np.mean((y==yt).astype('int64'))

def penalty_score(y, yt):
    s = np.zeros(shape=y.shape)
    for i in range(len(s)):
        # Correct Prediction
        if y[i] == yt[i]:
            s[i] = 0
        # Edible but labeled Poinsonous
        elif y[i] == 0:
            s[i] = 1
        # Poisonous but labeled Edible
        else:
            s[i] = 100
    return np.mean(s)
