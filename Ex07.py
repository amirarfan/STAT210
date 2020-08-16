# -*- coding: utf-8 -*-

__author__ = 'Amir Arfan'
__email__ = 'amir.inaamullah.arfan@nmbu.no'

## kgf/cm^2 som spesifisert i csv filen.

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
cement_data = pd.read_csv("PortlandCementData.csv", sep=";", decimal=",")
# print(cement_data.head())
cement_data = pd.melt(cement_data,var_name="Modification", value_name="Strength")
cement_data = cement_data.reindex(columns=["Strength", "Modification"])
# print(cement_data[["Modified", "Unmodified"]].mean())
# print(cement_data[["Modified", "Unmodified"]].std())
# print(cement_data[["Modified", "Unmodified"]].max())
# print(cement_data[["Modified", "Unmodified"]].min())
# quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
# print(cement_data[["Modified", "Unmodified"]].quantile(q=quantiles))
# cement_data.boxplot()
# plt.show()

mod = ols("Strength ~ C(Modification, Sum)", data=cement_data).fit()
print(mod.summary())
anov = sm.stats.anova_lm(mod)
print(anov)