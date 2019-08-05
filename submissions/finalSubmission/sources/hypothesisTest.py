from scipy.stats import t
import math

def t_test(mean1, std1, mean2, std2, n, alpha=0.05):
    # Calculate test params
    X = ((mean1-mean2)*math.sqrt(n))/math.sqrt(std1**2+std2**2)

    # Calculate degrees of freedom
    v = (std1**2 + std2**2)**2*(n-1)/(std1**4+std2**4)

    # Get x_{1-alpha, v}
    xAlpha = t.ppf(1-alpha,v)

    # Is null hypothesis rejected?
    return X > xAlpha
