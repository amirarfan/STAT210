from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects import Formula
from rpy2.robjects.vectors import StrVector
import math
import seaborn as sns
import scipy.stats as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.special import binom
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import numpy as np
import pprint as pp

my_data = pd.read_csv("SalmonLice.csv", sep=";", decimal=",", header=0)
my_mod = ols("SalmonLice ~ C(CleanerFish,Sum)/C(Farm, Sum)", data=my_data).fit()
est_mean = my_data["SalmonLice"].mean()
# print(my_mod.summary())
anov = sm.stats.anova_lm(my_mod) # For estimering av parametere, eller med mean måten
control_mean = my_data.loc[my_data["CleanerFish"] == "Control"]["SalmonLice"].mean()
lumpfish_mean = my_data.loc[my_data["CleanerFish"] == "Lumpfish"]["SalmonLice"].mean()
wrappers_mean = my_data.loc[my_data["CleanerFish"] == "Wrappers"]["SalmonLice"].mean()

t_1, t_2, t_3 = (control_mean - est_mean), (lumpfish_mean - est_mean), (wrappers_mean - est_mean)
# print(t_1, t_2, t_3)
# print()
# print(anov)

sigma_b = (anov.mean_sq["C(CleanerFish, Sum):C(Farm, Sum)"] - anov.mean_sq["Residual"])/8
sigma_e = anov.mean_sq["Residual"]
# print(sigma_b, sigma_e)


# d)
incorrect_mod = ols("SalmonLice ~ C(CleanerFish, Sum)*C(Farm,Sum)", data=my_data).fit()
anov_incorrect = sm.stats.anova_lm(incorrect_mod)
# print(anov_incorrect) # Can see that the SS for the treatment is the same,
# and the SSt can be calculated, however the mean square is difficult to
# calculate, thus the variance is difficult to calculate.

# e) H_0 : Alle tau = 0, H_A: Minst en tau ikke lik 0.
F_val = anov.mean_sq["C(CleanerFish, Sum)"] / anov.mean_sq["C(CleanerFish, Sum):C(Farm, Sum)"]
p_val = st.f.sf(F_val, 2, 3) # P_val < 0.05, Rejects H_0 og Cleanerfish har mest sannsynlig en effekt på signifikant nivå på 0.05.

# f) Test med Kontraster
#H_0 : Contrasts = 0 , H_A: Contrasts < 0
contrasts = np.array([1/2, 1/2, -1])
est_array = np.array([t_2, t_3, t_1])
contrast_val = sum(contrasts*est_array)
# print(contrast_val)
T_0 = contrast_val / (math.sqrt((anov.mean_sq["C(CleanerFish, Sum):C(Farm, Sum)"]/(8*2))*sum(contrasts**2)))
crit_t_val = st.t.ppf(1-0.05, 3)
# print(T_0, -1*crit_t_val)
# Avviser H_0

# g) By Increasing b, we would increase df for all variables. By increasing n we would only increase df for Residuals which has enough already (42). Better to increase fish farms.

# h) Varians hypotesetest, siden det er en random variabel. H_0 : Sigma^2_b = 0, H_A : Sigma^2_b != 0
F_val = anov.mean_sq["C(CleanerFish, Sum):C(Farm, Sum)"]/anov.mean_sq["Residual"]
crit_f_val = st.f.ppf(1-0.05, 3, 42)
p_val = st.f.sf(F_val, 3, 42)
print(F_val, crit_f_val, p_val) #H_0 er avvist.

