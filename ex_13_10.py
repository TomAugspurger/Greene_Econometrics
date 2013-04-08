from __future__ import division, print_function, unicode_literals

import pandas as pd

# Example 13.10 p. 503; 545e;

cols = ['id', 'year', 'expend', 'rev', 'grants']
df = pd.read_csv('T7987.asc', header=None, sep='\s+', names=cols,
                 index_col=['id', 'year'])

diffed = df.groupby(level='id').diff().dropna()
