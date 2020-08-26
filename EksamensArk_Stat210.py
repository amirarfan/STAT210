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
from statsmodels.stats.libqsturng import psturng, qsturng




## Anova - Fixed Model
a = 5
n = 10
N = 50
dftr = a - 1
dfe = N - a
SS_E = 4998.7
SS_T = 1708.2
MS_E = SS_E / dfe
MS_T = SS_T / dftr
F_val = MS_T / MS_E
alpha = 0.05
crit_f = stats.f.ppf(1-alpha, dftr, dfe)
p_value = stats.f.sf(F_val, dftr, dfe)
print(f"""
The F_value is: {F_val}
The Crit_F value: {crit_f}
Reject NullHypothesis: {F_val > crit_f}
P Value: {p_value}
""")

## Tukey Fixed Effect Model

# How many pairs
num_pairs = binom(5, 2)
print(num_pairs)

# Tukey's Test
#H_0 : Alle gjennomsnitt like, H_A: Minst et gjennomsnitt som ikke er likt.
T_val = qsturng(1-0.05, 5, 45)*math.sqrt(MS_E/n)
print(T_val)
mean_array = np.array([68.6, 58.4, 58.6, 64.4, 73.5])
diff_array = [round(abs(x - y), 2) for i, x in enumerate(mean_array) for j, y in enumerate(mean_array) if i != j]
print(diff_array)
print(diff_array > T_val)
# Ut i fra denne kan man se at M2-M5 og M3-M5 har signifikant forskjeller.

## Intraclass Correlation
sigma_t = (MS_T - MS_E)/ n
sigma_e = MS_E
intraclass_corr = (sigma_t)/(sigma_t +sigma_e)
print(intraclass_corr)
# 0.22, which says that 22% of the variation is explained by the Municpaility

# Konfidens-Intervall for Intraclass Correlation
L = 1/n * (((MS_T/MS_E)*(1/stats.f.isf(0.025, dftr, dfe))) - 1)
U = 1/n * (((MS_T/MS_E)*(1/stats.f.isf(0.975, dftr, dfe))) - 1)

lower = L/(1+L)
upper = U/(1+U)
print(lower, upper) # 95% sannsynlighet at intervallet består av den sanne korrelasjonen. 0.22 virker veldig usikkert.


# Hypotesetest for Kontraster
contrasts = [1, -1]
sum_of_contrasts_mean = contrasts[0]*5.48 + contrasts[1]*5.07
sum_of_contrasts_sq = (contrasts[0])**2+(contrasts[1])**2
MS_E = 0.8576/50
n= 6
T_0 = 0.262
T_val = (sum_of_contrasts_mean - T_0)/math.sqrt((MS_E/n)*sum_of_contrasts_sq)
t_crit = stats.t.ppf(1-0.05, 50)
print(T_val, t_crit)
# Siden T_val > t_crit, avviser H_0



