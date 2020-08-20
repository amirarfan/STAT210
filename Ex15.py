import math
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from scipy.special import binom
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import numpy as np
import pprint as pp

# a)
not_clever_data = pd.read_csv("BeerDataNotClever.csv", sep=";", header=0, index_col=0)
tab_1 = not_clever_data.groupby(["Ingredience1", "Ingredience2", "Ingredience3"]).size()
# pp.pprint(tab_1.unstack())


clever_data = pd.read_csv("BeerDataClever.csv", sep=";", header=0, index_col=0)
tab= clever_data.groupby(["Ingredience1", "Ingredience2", "Ingredience3"]).size()
# pp.pprint(tab.unstack())

# Only the data from Clever Student can be used for a complete model

clever_mod = ols("Taste ~ C(Ingredience1, Sum)*C(Ingredience2, Sum)*C(Ingredience3, Sum)", data=clever_data).fit()
# print(clever_mod.summary())

not_clever_mod = ols("Taste ~ C(Ingredience1, Sum)*C(Ingredience2, Sum)*C(Ingredience3, Sum)", data=not_clever_data).fit()
# print(not_clever_mod.summary()) # Can see that the design matrix is singular, i.e not good fit

# c)
full_anov_clev = sm.stats.anova_lm(clever_mod)
# print(full_anov_clev) # P-value suggests that there is little interaction between the Ingredience 1 terms
reduced_clever_mod = ols("Taste ~ C(Ingredience2, Sum)*C(Ingredience3, Sum)", data=clever_data).fit()
reduced_anov_clev = sm.stats.anova_lm(reduced_clever_mod)
# print(reduced_anov_clev)

# Partial F-test
# H_0 : Ingredience 1 has no effect
# H_A : Ingredience 1 has effect
sser = reduced_anov_clev.sum_sq["Residual"]
ssef = full_anov_clev.sum_sq["Residual"]
dfer = reduced_anov_clev.df["Residual"]
dfef = full_anov_clev.df["Residual"]

partial_F = ((sser - ssef)/ (dfer - dfef)) / (ssef / dfef)
# print(partial_F)
crit_f_value = stats.f.ppf(1-0.05, dfer-dfef, dfef)
# print(crit_f_value)
# print(partial_F > crit_f_value) # -> Accept H_0

# Thus we use the reduced model, now we can test the model assumptions
# Check for constant variance
plt.figure(figsize=(8, 5))
scat = sns.scatterplot(x=reduced_clever_mod.fittedvalues, y=reduced_clever_mod.resid)
xmin=min(reduced_clever_mod.fittedvalues)
xmax = max(reduced_clever_mod.fittedvalues)
plt.hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
plt.xlabel("Fitted values",fontsize=13)
plt.ylabel("Residuals",fontsize=13)
plt.title("Fitted vs. residuals plot",fontsize=19)
plt.grid(True)
# plt.show()
# Seems to be constant variance and randomly distributed, but small sample

# Check for Normality
 # Check for Normality
plt.figure(figsize=(8,5))
fig=qqplot(reduced_clever_mod.resid_pearson,line='45',fit='True')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
# plt.show()

# Seems to be Normal

# g)
not_clever_mod = ols("Taste ~ C(Ingredience1, Sum)+C(Ingredience2, Sum)+C(Ingredience3, Sum)", data=not_clever_data).fit()
print(not_clever_mod.summary())

