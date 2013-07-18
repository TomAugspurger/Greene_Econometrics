from numpy.testing import assert_
from ..example_13_7 import GMM


def check_final_answer():
    import numpy as np
    fldr = '/Users/tom/Dropbox/Economics/Econometrics_2/Greene_Econometrics/'
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

    dta_gmm = df[['hhninc', 'const', 'age', 'educ', 'female', 'hsat',
                  'married']].xs(1988, level='year')

    dta_gmm = dta_gmm[dta_gmm.hhninc > 0]

    endog = ['hhninc']
    exog = ['const', 'age', 'educ', 'female']
    instruments = ['hsat', 'married']

    obj = GMM(dta_gmm, endog, exog, instruments)
    res = obj.fit_exp([-1, 1, .1, .2])
    expected = np.array([7.4575886876535966, 0.0017542784300367533,
                         0.053972808533195776, 0.036516820396588774])
    assert_(res == expected)
