# p. 502e; p. 460


# Gamma distribution
# Table FC1

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gamma, gammaln, digamma
from scipy import stats


df = pd.read_csv('http://www.stern.nyu.edu/~wgreene/'
                 'Text/Edition7/TableFC-1.csv',
                 index_col='I')


def mom_cond(theta, x=df.Y.values):
    P, lambda_ = theta
    m1 = x - P / lambda_
    m2 = x ** 2 - (P * (P + 1) / lambda_**2)
    m3 = np.log(x) - (digamma(P) - np.log(lambda_))
    m4 = (1 / x) - lambda_ / (P - 1)
    return np.vstack([m1, m2, m3, m4]).T.sum(0)  # k x 1

sample_moms = (np.array([df.Y, df.Y ** 2, np.log(df.Y), 1 / df.Y]).sum(1) /
               len(df.Y))


def pick_moms(theta, x, l=[0, 1]):
    """
    List of pairs used to estimate.
    """
    return sum(mom_cond(theta, x)[l] ** 2)

"""
I've got the estimation somewhat.

    optimize.fmin(pick_moms, x0=[2, 2], args=[df.Y.values, [3, 3]])

"""
for pair in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
    print('{}\n'.format(pair))
    print(optimize.fmin(pick_moms, x0=[2, 2], args=[df.Y.values, pair]))
    print('\n\n')

# all of them
optimize.fmin(pick_moms, x0=[2, 2], args=[df.Y.values, [0, 1, 2, 3]])
