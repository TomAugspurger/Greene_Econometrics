import numpy as np
from numpy import dot
import pandas as pd
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
from scipy import optimize
from scipy.linalg import inv
import pdb
# qed.econ.queensu.ca/jae/2003-v18.4/riphahn-wambach-million/rwm-data.zip
# qed.econ.queensu.ca/jae/2003-v18.4/riphahn-wambach-million/readme.rwm.txt


class GMM(object):
    """
    Earns a class.

    example
        endog = ['hhninc']
        exog = ['const', 'age', 'educ', 'female']
        instruments = ['hsat', 'married']

        obj = GMM(dta_gmm, endog, exog, instruments)
        res = obj.fit_exp([-1, 1, .1, .2])
    """
    def __init__(self, data, endog, exog, instruments, method='Nelder-Mead', form='linear'):
        """
        data: An entire dataframe
        endog: Str. Column name.
        exog: List: exogenous variables.
        instruments: list.  Doesn't include exog.

        Probably want to add a formula option and integrate
        with patsy.
        """
        self.data = data
        # if set.intersection(set(exog), set(instruments)) != set([]):
        #     raise ValueError('Don\'t put cols in exog and instruments.')
        self.endog = endog
        self.exog = list(exog)
        try:
            self.instruments = list(instruments)
            self.generalized = True
            self.n_moms = len(self.exog) + len(self.instruments)
        except TypeError:
            self.instruments = None
            self.generalized = False
            self.n_moms = len(self.exog)

        self.n = len(data[endog])
        self.form = form
        self.method = method

    def mom_gen_exp(self, theta):
        """
        theta: initial guess
        """
        data = self.data
        theta = np.array(theta).reshape(len(self.exog), 1)
        y = data[self.endog].values
        X = data[self.exog].values
        e = y - np.exp(dot(X, theta))

        if self.generalized:
            Z = data[self.exog + self.instruments].values
            m = dot(Z.T, e) / self.n
            # pdb.set_trace()
            try:
                moments = dot(dot(m.T, self.W), m)
            except (TypeError, AttributeError):
                moments = dot(m.T, m)
        else:
            m = dot(X.T, e) / self.n
            moments = dot(m.T, m)
        pdb.set_trace()
        return moments

    def fit_exp(self, x0, maxiter=None):
        """
        Greene p. 487 for weight matrix.
        """
        x0 = np.array(x0).reshape(len(self.exog), 1)
        round_one = optimize.minimize(self.mom_gen_exp, x0=x0,
                                      method=self.method,
                                      options={'maxiter': maxiter})
        ### Now Solve for Optimal W
        # Greene p. 490; Check if this is right.
        # Using White's (1980) estimator.
        round_one = round_one['x'].reshape(len(self.exog), 1)
        y = self.data[self.endog].values
        X = self.data[self.exog].values
        e = y - np.exp(dot(X, round_one))

        if self.generalized:
            Z = self.data[self.exog + self.instruments].values

            for i, obs in enumerate(Z):
                zi = obs.reshape(Z.shape[1], -1)
                if i == 0:
                    W = dot(zi, zi.T) * (e[i]) ** 2
                else:
                    W += dot(zi, zi.T) * (e[i]) ** 2
            self.W = inv(W / self.n)
            # round_two implicitly uses W.  Probably a better way.
            # This also caches W forever until explicity removed.
            round_two = optimize.fmin(self.mom_gen_exp, x0=round_one,
                                      maxiter=maxiter)
            return round_two
        else:
            return round_one

if __name__ == '__main__':
    fldr = '/Users/tom/Dropbox/Economics/Econometrics_2/Greene_Econometrics/'
    cols = ['id', 'female', 'year', 'age', 'hsat', 'handdum', 'handper',
            'hhninc', 'hhkids', 'educ', 'married', 'haupts', 'reals', 'fachhs',
            'abitur', 'univ', 'working', 'bluec', 'whitec', 'self', 'beamt',
            'docvis', 'hospvis', 'public', 'addon']

    df = pd.read_csv(fldr + 'data/rwm.data', names=cols, sep='\s+',
                     index_col=['id', 'year'])
    df['const'] = 1
    # model: income ~ exp(age + educ + female)

    #################################################
    #            Part One: Method of Moments.       #
    #            Basically just an orthogonality    #
    #            condition for the regressors.      #
    #################################################
    dta = df.xs(1988, level='year')
    dta = dta[dta.hhninc > 0]  # 2 houses with zero income.
    sub_cols = ['hhninc', 'const', 'age', 'educ', 'female']
    dta = dta[sub_cols]

    obj = GMM(dta, ['hhninc'], ['const', 'age', 'educ', 'female'], None, form='exp')
    res = obj.fit_exp([-1.69331, .00207, .04792, -.00658], maxiter=2000)
    print(res)

    #####################################################
    #        Part Two: Generalize Method of Moments.    #
    #        Additional orthogonality requirements      #
    #        for hsat and married                       #
    #####################################################

    dta_gmm = df[['hhninc', 'const', 'age', 'educ', 'female', 'hsat',
                  'married']].xs(1988, level='year')

    dta_gmm = dta_gmm[dta_gmm.hhninc > 0]

    endog = ['hhninc']
    exog = ['const', 'age', 'educ', 'female']
    instruments = ['hsat', 'married']

    obj = GMM(dta_gmm, endog, exog, instruments, method='Nelder-Mead', form='exp')
    res = obj.fit_exp([-1, 1, .1, .2], maxiter=2000)
    print(res)
