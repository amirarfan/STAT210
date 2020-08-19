import math
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import binom
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import numpy as np

my_data = pd.read_csv("Beefcarcasses.csv", sep=";", header=0, decimal=",")
# print(my_data)
# print(my_data.groupby(["Breed", "Gender"]).count())
# Ser at Dataen er Balansert

# H_0 : (tau*beta)ij = 0,    H_A: (tau*beta)ij != 0
# formula = "KFactor ~ C(Breed, Sum)*C(Gender, Sum)"
# mod = ols(formula, data=my_data).fit()
# print(mod.summary())
# anov = sm.stats.anova_lm(mod)
# print(anov)
# Aksepterer H_0 da p-verdi mellom interaksjons faktorene er = 0.36
# C ut i fra tabellen.

# d)

# fig, ax = plt.subplots(figsize=(6, 6))
# fig = interaction_plot(x=my_data["Breed"], trace=my_data["Gender"], response=my_data["KFactor"],
#                        colors=['red', 'blue'], markers=['D', '^'], ms=10, ax=ax)
# plt.show()

# e) Model without interaction, i.e interaction factor is removed
formula = "KFactor ~ C(Breed, Sum)+C(Gender, Sum)"
mod = ols(formula, data=my_data).fit()
anov = sm.stats.anova_lm(mod)
# print(anov)
# f) SS_EM = SS_ABC + SS_EC

# g) DF øker og kompenserer for den økte SS_E (MS = SS/df)

# h) Contrasts= -1 * Holstein + 0 * Limousine + 1 * NRF
# H_0 : Sum of Contrasts = 0, H_A : Sum of Contrasts != 0
contrasts = np.array([-1,0,1])
coeffs = np.array([-0.5129, 0.7857, -0.2728]) # Breed1 + Breeds2 - Breed3 = 0 # Må finne den siste på den måtpen
# print(coeffs)
beta = sum(coeffs*contrasts)
se_val = math.sqrt((anov.mean_sq["Residual"]/12)*(sum(x ** 2 for x in coeffs)))
T_val = beta/se_val
crit_t_val = stats.t.ppf(1-0.25, df=68)
# print(T_val)
# print(crit_t_val)
# p_val = stats.t.sf(T_val, 68)
# print(p_val)
# Får at Beta != 0 , regner ut T0 og finner ut at vi kan avvise H_0 og konkludere med at NRF har høyere gjennopmsnitt en KFactor og Holstein


#i )
contrasts = np.array([-1/2, 1, -1/2])
coeffs = np.array([-0.5129, 0.7857, -0.2728]) # Breed1 + Breeds2 - Breed3 = 0 # Må finne den siste på den måtpen
beta = sum(coeffs*contrasts)
# ... Konkluderer med at H_0 er avvist, Limousine er høyere gjennomsnittlig enn de to andre.

# j) Tukeys Test
comparisons = MultiComparison(my_data["KFactor"],my_data["Breed"])
print(comparisons.tukeyhsd(alpha=0.05).summary())

#k ) Limousine skiller seg fra de to andre gruppene, ser at Tukeys skiller seg fra H, det kreves større forskjell for Tukeys.




