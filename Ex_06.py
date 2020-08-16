# -*- coding: utf-8 -*-

__author__ = 'Amir Arfan'
__email__ = 'amir.inaamullah.arfan@nmbu.no'

## Exercise 06

import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm

# SoyaTest = pd.read_clipboard(sep=" ")
# print(SoyaTest.head())

# SoyaStacked = SoyaTest.stack()

## C)
soya_data = pd.read_csv("SoyaData.csv", header=0, sep=";")
print(soya_data.head())

mod = ols("Weight~Diet", data=soya_data).fit()
anov = sm.stats.anova_lm(mod)
# print(anov)

soya_data["Weight"] = [80, 84, 88, 90, 88, 92]
mod = ols("Weight~Diet", data=soya_data).fit()
anov = sm.stats.anova_lm(mod)
print(anov)