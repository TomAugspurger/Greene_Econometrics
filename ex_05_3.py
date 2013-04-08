import pandas as pd
import statsmodels.formula.api as sm
url = 'http://www.stern.nyu.edu/~wgreene/Text/Edition7/TableF5-2.txt'

df = pd.read_csv('http://www.stern.nyu.edu/~wgreene/Text/Edition7/TableF5-2.txt',
                 sep='\s+')
df['time'] = df.Year + df.qtr / 4


m1 = sm.ols(formula='np.log(realinvs) ~ tbilrate + infl + np.log(realgdp) + time',
            data=df)

r1 = m1.fit()
print(r1.summary())
