# -*- coding: utf-8 -*-

__author__ = 'Amir Arfan'
__email__ = 'amir.inaamullah.arfan@nmbu.no'

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
my_df = pd.read_csv("SiresData.csv", sep=";", header=0)
mod = ols("Milk ~ C(Sire, Sum)", data=my_df).fit()
anov = sm.stats.anova_lm(mod)
print(anov)
print(mod.summary())