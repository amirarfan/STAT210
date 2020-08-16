# -*- coding: utf-8 -*-

__author__ = 'Amir Arfan'
__email__ = 'amir.inaamullah.arfan@nmbu.no'

import pandas as pd
import statsmodels.api as sm
from scipy import stats
import math
from statsmodels.formula.api import ols

my_df = pd.read_csv("SiresData.csv", sep=";", header=0)
mod = ols("Milk ~ C(Sire, Sum)", data=my_df).fit()
anov = sm.stats.anova_lm(mod)
# print(mod.summary())

# Confidence Interval for Variance

alpha = 0.05
ss_e = anov.sum_sq["Residual"]
# print(ss_e)
conf_inter = ss_e / stats.chi2.isf(q=(alpha/2, 0.975), df=35)
# print(conf_inter)


# e)
# H_0 = tau_1 = 150, H_A = tau_1 > 150
# Benytter oss av en T-test
# Finner tau_1 fra model summary
tau_1 = 171.10
prop_tau_1 = 150
est_se = 105.58
df = 35
t_value = stats.t.ppf(0.9, 35)
T_test = (tau_1 - prop_tau_1) / est_se
# print(t_value)
# print(T_test > t_value)
# Konkluderer med at H_0 kan ikke bli avvist.


# f)
# H_0 = mu_4 = 6500, H_A = mu_4 > 6500
# Bruker en T-test
ms_e = anov.mean_sq["Residual"]
est_se = math.sqrt(ms_e/8)
T_test = ( 6518.15+152.97 - 6500 ) / est_se
t_value = stats.t.ppf(0.95, 35)
# print(T_test > t_value)
# Konkluderer med at H_0 kan ikke avvises

# g)
# For å avvise H_0 må T verdien være større en kritisk t-verdi
# Finne kritisk t-verdi
t_value = stats.t.ppf(0.95, 35)
max_mean = 6518.15+152.97 - t_value*est_se
# print(max_mean)

# h)
# H_0 = mu_4 = mu_3, H_A = mu_3 != mu_4
# Siden V(mu_4 - mu_3) = V(mu_4) + V(mu_3) = 2*V(mu))
est_se = math.sqrt(2*ms_e/8)
T_test = ((6518.15+ 152.97) - (6518.15 - 192.40)) / est_se
t_value = stats.t.ppf(0.975, 35)
print(T_test > t_value)
# H_0 kan bli avvist på 5% signifikans





