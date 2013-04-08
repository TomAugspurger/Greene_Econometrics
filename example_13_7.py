from collections import OrderedDict

import numpy as np
from numpy import dot
import pandas as pd
import matplotlib.pyplot as plt
# import statsmodels.api as sm
from scipy import optimize
from scipy.linalg import inv
# qed.econ.queensu.ca/jae/2003-v18.4/riphahn-wambach-million/rwm-data.zip
# qed.econ.queensu.ca/jae/2003-v18.4/riphahn-wambach-million/readme.rwm.txt
fldr = '/Users/tom/Dropbox/Economics/Econometrics_2/greene_examples/'
cols = ['id', 'female', 'year', 'age', 'hsat', 'handdum', 'handper',
        'hhninc', 'hhkids', 'educ', 'married', 'haupts', 'reals', 'fachhs',
        'abitur', 'univ', 'working', 'bluec', 'whitec', 'self', 'beamt',
        'docvis', 'hospvis', 'public', 'addon']

df = pd.read_csv(fldr + 'data/rwm.data', names=cols, sep='\s+',
                 index_col=['id', 'year'])
df['const'] = 1
# model: income ~ exp(age + educ + female)

dta = df.xs(1988, level='year')
dta = dta[dta.hhninc > 0]  # 2 houses with zero income.
sub_cols = ['hhninc', 'const', 'age', 'educ', 'female']
dta = dta[sub_cols]


def mm_sse(theta, data):
    y, const, age, educ, female = data
    n = len(y)

    e = y - np.exp(dot(data[1:, :].T, theta))
    m1 = sum(e)
    m2 = dot(e, age)
    m3 = dot(e, educ)
    m4 = dot(e, female)
    m = np.hstack([m1, m2, m3, m4]) / n
    return sum(m ** 2)

x0 = [-1.0, .002, .05, .013]
res_fmin = optimize.fmin(mm_sse, x0=x0, args=[dta.values.T])

methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
           'Anneal', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

res_mm = {}
for method in methods:
    try:
        res_mm[method] = optimize.minimize(
            mm_sse, x0=x0, method=method, args=[dta.values.T])
    except:
        pass
d = OrderedDict(res_mm)
df_res_gmm = pd.DataFrame([x[1] for x in d.iteritems()], index=d.keys())

##### Now checking with GMM, 2 additional momen conditions.

dta_gmm = df[['hhninc', 'const', 'age', 'educ', 'female', 'hsat',
              'married']].xs(1988, level='year')

dta_gmm = dta_gmm[dta_gmm.hhninc > 0]


def gmm_sse(theta, data):
    y, const, age, educ, female, hsat, married = data
    n = len(y)

    e = y - np.exp(dot(data[1:5, :].T, theta))
    m1 = sum(e) / n
    m2 = dot(e, age) / n
    m3 = dot(e, educ) / n
    m4 = dot(e, female) / n
    m5 = dot(e, hsat) / n
    m6 = dot(e, married) / n
    m = np.hstack([m1, m2, m3, m4, m5, m6])
    return sum(m ** 2)


def gmm_sse2(theta, data):
    y, const, age, educ, female, hsat, married = data
    n = len(y)
    e = y - np.exp(dot(data[1:5, :].T, theta))
    m1 = sum(e)
    m2 = dot(e, age)
    m3 = dot(e, educ)
    m4 = dot(e, female)
    m5 = dot(e, hsat)
    m6 = dot(e, married)
    z = data[1:]
    m = np.hstack([m1, m2, m3, m4, m5, m6])
    el1 = dot(e, z.T).reshape(6, 1)
    W = dot(el1, el1.T) / n
    return dot(dot(m.T / n, inv(W)), m / n)


def mom_gen(theta, data, *args):
    """
    Refactor of sse2 but using matrix operations and splitting the
    operation into two parts:
        1. mom_gen: generate the moment conditions.
        2. min_obj: minimize the objective function.

    Generalizes to having *every* column in the data being orthogonal
    to the erros.
    Careful; I'm getting the shapes right so I pass data.values instead of
    data.values.T into args=[].
    """
    y = data[:, 0]
    n = len(y)
    X = data[:, 1:5]
    Z = data[:, 1:]
    e = y - np.exp(dot(X, theta))

    m = np.array([dot(e, x) for x in Z.T]) / n
    # el1 = dot(e, z.T).reshape(6, 1)
    # W = dot(el1, el1.T) / n
    # return dot(dot(m.T / n, inv(W)), m / n)
    try:
        W = args[0]
    except IndexError:
        W = np.eye(len(m))

    return dot(dot(m.T, W), m.T)

# def sseify(fn, *args, **kwargs):
#     """
#     Take a function.  Return the sum of squared errors version
#     of that function.
#     """
#     return dot(fn(x, args, kwargs).T, fn(x, args, kwargs))


def min_obj(fn, data, x0=x0, **kwargs):
    """
    A bit higher level.  Pass it some kind of moment generating
    function (but one that actually returns the SSE (need to think
    about how to raise the moments up on the final run)).

    Example:
        min_obj(mom_gen, dta_gmm, x0=t_mm_s)
    """
    Z = data.values[:, 1:]
    W = inv(dot(Z.T, Z))
    t_hat_one = optimize.fmin(mom_gen, x0=x0, args=[data.values, W])
    return t_hat_one


def gmm_sse2_mat2(theta, data):
    """
    May be all wrong if this only applies to linear case...
    Refactor of sse2 but using matrix operations.

    Generalizes to having *every* column in the data being orthogonal
    to the erros.

    Following Greene p. 484
    Right now also has the estimator from 484, but not returned.
    """
    y = data[0, :]
    n = len(y)
    X = dta_gmm.values.T[1:5, :].T
    Z = dta_gmm.values.T[1:, :].T

    m = (1 / n) * dot(Z.T, y) - (1 / n) * dot(dot(Z.T, X), theta)  # LINEAR!
    e = y - np.exp(dot(X, theta))
    b = dot(inv(dot(dot(X.T, Z), dot(Z.T, X))), dot(dot(X.T, Z), dot(Z.T, y)))
    return (m, e, b)


def step_1(theta, data):
    y = data[0, :]
    n = len(y)
    X = dta_gmm.values.T[1:5, :].T
    Z = dta_gmm.values.T[1:, :].T

    m = (1 / n) * dot(Z.T, y) - (1 / n) * dot(dot(Z.T, X), theta)
    e = y - np.exp(dot(X, theta))



res_gmm = {}
for method in methods:
    try:
        res_gmm[method] = optimize.minimize(
            gmm_sse, x0=x0, method=method, args=[dta_gmm.values.T])
    except:
        pass

d = OrderedDict(res_gmm)
df_res_gmm = pd.DataFrame([x[1] for x in d.iteritems()], index=d.keys())


def part_plot(fn, x0, free, free_range, data):
    """A little helper to plot obj function over a range for one
    parameter, holding others fixed.  Not working great right now.
    The gradient needs to be zero at the maximum right?
    """
    sol = optimize.fmin(fn, x0=x0, args=[data.values.T])
    n = len(free_range)
    theta_range = np.tile(sol, n).reshape(n, -1)
    theta_range[:, 0] = free_range
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(free_range, [fn(y, data.values.T) for y in theta_range])
    return ax


### Trying to generalize into r-style from here on
def gg(theta, data):
    """
    """
    y, const, age, educ, female = data
    n = len(y)

    e = y - np.exp(dot(data[1:, :].T, theta))
    m1 = sum(e) / n
    m2 = dot(e, age) / n
    m3 = (1.0 / n) * dot(e, educ)
    m4 = (1.0 / n) * dot(e, female)
    f = np.vstack([m1, m2, m3, m4]).T
    return f


def min_(x0=[1, 1, 1, 1], method='Nelder-Mead'):
    res = optimize.minimize(mm_sse, x0=x0, method=method, args=dta.values.T)
    return res


def gmm_gg(theta, data):
    y, const, age, educ, female, hsat, married = data
    n = len(y)

    e = y - np.exp(dot(data[1:5, :].T, theta))
    m1 = (1.0 / n) * sum(e)
    m2 = (1.0 / n) * dot(e, age)
    m3 = (1.0 / n) * dot(e, educ)
    m4 = (1.0 / n) * dot(e, female)
    m5 = dot(e, hsat) / n
    m6 = dot(e, married) / n
    return np.vstack([m1, m2, m3, m4, m5, m6]).T


t_mm_s = np.array([-1.62969, 0.00178, 0.04861, 0.01384])

##############################################################################
