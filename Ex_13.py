import math
import pingouin as pg
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from scipy.special import binom
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import numpy as np


my_data = pd.read_csv("SiresData.csv", sep=";", header=0)
mod = ols("Milk ~ C(Sire, Sum)", data=my_data).fit()
anov = sm.stats.anova_lm(mod)
#H_0: Variansen mellom treatmentsa er lik 0, H_A: Variansen er > 0, signifikantverdi 0.05
print(anov)
# Ut ifra P-verdien avvises H_0, og man kan si at variansen er ikke lik 0, og det er effekt fra oksene
#c) oppgaven
print(mod.summary())
print()
# mean = 6518.15
se_tr = (anov.mean_sq["C(Sire, Sum)"] - anov.mean_sq["Residual"])/8
print(se_tr)

# d) Confidence Interval for Error Variance
n = 8
sse = anov.sum_sq["Residual"]
df = 35
alpha = 0.05

upper = sse / stats.chi2.ppf(alpha / 2, df)
lower = sse / stats.chi2.ppf(1 - alpha / 2, df)

print(lower, upper)

# Confidence interval for Mean
n = 8
a = 5
alpha = 0.05
df = 35
est_mean = mod.params[0]
lower = est_mean - (stats.t.isf(alpha/2, 4)*math.sqrt(anov.mean_sq["C(Sire, Sum)"]/(a*n)))
upper = est_mean + (stats.t.isf(alpha/2, 4)*math.sqrt(anov.mean_sq["C(Sire, Sum)"]/(a*n)))
# print(lower, upper)

#e) Intraclass Correlation
corr = se_tr / (se_tr + anov.mean_sq["Residual"])
# print(corr)

# Intrclass Confidence Interval
L = 1/n * (((anov.mean_sq["C(Sire, Sum)"]/anov.mean_sq["Residual"])*(1/stats.f.isf(0.05, 4, 35))) - 1)
U = 1/n * (((anov.mean_sq["C(Sire, Sum)"]/anov.mean_sq["Residual"])*(1/stats.f.isf(0.95, 4, 35))) - 1)

lower = L/(1+L)
upper = U/(1+U)

print(lower, upper)

