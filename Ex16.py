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

# Svar på a)
# Det blir 1 overall mean, 4 treatment effects, 6 2-faktor , 4- 3 faktor og 1 4 faktor
# b)
# df_e = abcd(n-1)  = 16 i dette tilfellet
#c)
my_data = pd.read_csv("Barley.csv", sep=";", decimal=",", header=0)
tab = my_data.groupby(["Variety", "Soil", "Fertilizer", "Site"]).size()
# print(tab.unstack())
# yes it does
#d)
# print(my_data.head())
my_mod = ols("Yield ~ C(Variety, Sum)*C(Soil, Sum)*C(Fertilizer, Sum)*C(Site, Sum)",  data=my_data).fit()
anov = sm.stats.anova_lm(my_mod)
reduced_mod = ols("Yield ~ C(Variety, Sum)*C(Soil, Sum)*C(Fertilizer, Sum)", data=my_data).fit()
reduced_anov = sm.stats.anova_lm(reduced_mod)
# Partial F_Test
# H_0: Site has no Effect
# H_A: Site has an effect
sser = reduced_anov.sum_sq["Residual"]
ssef = anov.sum_sq["Residual"]
dfer = reduced_anov.df["Residual"]
dfef = anov.df["Residual"]

partial_F = ((sser - ssef)/ (dfer - dfef)) / (ssef / dfef)
# print(partial_F)
crit_f_value = stats.f.ppf(1-0.05, dfer-dfef, dfef)
# print(crit_f_value)
p_value = stats.f.sf(partial_F, dfer-dfef, dfef)
# print(p_value)
# From p-value, one accepts the H0, and can use the reduced model
#f)
# print(reduced_anov)
# Based on 0.01 significance level one can skip the three interaction term.
more_reduced_mod = ols("Yield ~ (C(Variety, Sum)+C(Soil, Sum)+C(Fertilizer, Sum))**2", data=my_data).fit()
more_reduced_aov = sm.stats.anova_lm(more_reduced_mod)
# print(more_reduced_aov)
# One can see that Variety and fertilizer can be removed as well
final_reduced_mod = ols("Yield ~ (C(Variety, Sum)+C(Soil, Sum)+C(Fertilizer, Sum))**2-(C(Variety, Sum):C(Fertilizer, Sum))", data=my_data).fit()
final_reduced_aov = sm.stats.anova_lm(final_reduced_mod)
# print(final_reduced_aov) # Can't see any terms to reduce now with significant level of 0.01

#g)
# Divide the data
Barley_F1 = my_data[my_data.Fertilizer == "Fert1"]
Barley_F2 = my_data[my_data.Fertilizer == "Fert2"]
Barley_F1 = Barley_F1.reset_index()
Barley_F2 = Barley_F2.reset_index()
fig, axs = plt.subplots(nrows=2, figsize=(6, 6))
axs[0].set_title("Fertilizer Type 1 Interaction Plot")
axs[1].set_title("Fertilizer Type 2 Interaction Plot")
interaction_plot(x=Barley_F1["Soil"], trace=Barley_F1["Variety"], response=Barley_F1["Yield"], ax=axs[0])
interaction_plot(x=Barley_F2["Soil"], trace=Barley_F2["Variety"], response=Barley_F2["Yield"], ax=axs[1])
plt.show()