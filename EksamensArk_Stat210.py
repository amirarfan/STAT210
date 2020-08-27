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
a = 3
n = 21
N = a*n
dftr = a - 1
dfe = N - a
SS_E = 232000
SS_T = 60227
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

## Anova - Random Model
#H_0 : Sigma_t^2 = 0, H_A: Sigma_t^2 != 0
MS_Tr = 4.27
MS_E = 3.87
F_Val = MS_Tr / MS_E
p_val = stats.f.sf(F_val, 2, 12)
print(p_val)

# Confidence Interval - Variance term ( Error )
SS_E = 455.31
lower_chi = stats.chi2.isf(0.05/2, 12) # Merk at dette er 95% konfidensintervall
upper_chi = stats.chi2.isf(1 - (0.05/2), 12)
lower, upper = SS_E/lower_chi, SS_E/upper_chi
## Tukey Fixed Effect Model

# How many pairs
num_pairs = binom(5, 2)
# print(num_pairs)

# Tukey's Test
#H_0 : Alle gjennomsnitt like, H_A: Minst et gjennomsnitt som ikke er likt.
T_val = qsturng(1-0.05, 5, 45)*math.sqrt(MS_E/n)
# print(T_val)
mean_array = np.array([68.6, 58.4, 58.6, 64.4, 73.5])
diff_array = [round(abs(x - y), 2) for i, x in enumerate(mean_array) for j, y in enumerate(mean_array) if i != j]
# print(diff_array)
# print(diff_array > T_val)
# Ut i fra denne kan man se at M2-M5 og M3-M5 har signifikant forskjeller.

## Intraclass Correlation
MS_T = 817.07
MS_E = 81.83
n = 6
sigma_t = (MS_T - MS_E)/ n
sigma_e = MS_E
intraclass_corr = (sigma_t)/(sigma_t +sigma_e)
# print(intraclass_corr)
# 0.22, which says that 22% of the variation is explained by the Municpaility

# Konfidens-Intervall for Intraclass Correlation
dftr = 3-1
dfe= 5*3 - 3
L = 1/n * (((MS_T/MS_E)*(1/stats.f.isf(0.025, dftr, dfe))) - 1)
U = 1/n * (((MS_T/MS_E)*(1/stats.f.isf(0.975, dftr, dfe))) - 1)

lower = L/(1+L)
upper = U/(1+U)
# print(lower, upper) # 95% sannsynlighet at intervallet består av den sanne korrelasjonen. 0.22 virker veldig usikkert.


# Hypotesetest for Kontraster
contrasts = [1, -1]
sum_of_contrasts_mean = contrasts[0]*5.48 + contrasts[1]*5.07
sum_of_contrasts_sq = (contrasts[0])**2+(contrasts[1])**2
MS_E = 0.8576/50
n= 6
T_0 = 0.262
T_val = (sum_of_contrasts_mean - T_0)/math.sqrt((MS_E/n)*sum_of_contrasts_sq)
t_crit = stats.t.ppf(1-0.05, 50)
# print(T_val, t_crit)
# Siden T_val > t_crit, avviser H_0

# Hypotesetest for Nested Design Fixed-Random (Tester Fixed Parameter)
#H_0 : t_1 = t_2 = t_3 = 0, H_A: Minst en tau != 0.
a = 3
b = 2
MS_A = 7.46
MS_BA = 0.49
F_val = MS_A / MS_BA
p_val = stats.f.ppf(1-0.05, a-1, a*(b-1))
crit_f_val = stats.f.sf(F_val, a-1, a*(b-1))
# print(F_val, p_val, crit_f_val) # Avviser H_0.

## Random Block Design
SS_T = 306.1 + 15413.2 + 502.6
SS_E = 502.6
R_squared = 1 - (SS_E/SS_T)
# print(f"{round(R_squared,2)*100}% of the variation in cholesterol is explained by the model")

# Hypothesis test
# H_0: tau_1 = tau_2 = tau_3 = 0, H_A: tau_i != 0
SS_tr = 306.1
df_tr = 2
MS_tr = SS_tr/df_tr
MS_er = 502.6/18
F_val = (MS_tr/MS_er)
p_val = stats.f.sf(F_val, 2, 18)
# print(p_val) # H_0 is rejected on a significant level of 5%

# Contrast Test
#H_0 : T = T_0, H_A : T > 0
MS_er = 3866.7
contrasts = np.array([1/2, 1/2, -1])
mean_array = np.array([200.9, 180.2, 253.7])
sum_of_contrasts_mean = sum(contrasts*mean_array)
sum_of_contrasts_sq = sum(contrasts**2)
T_0 = (sum_of_contrasts_mean)/math.sqrt((MS_er/21)*sum_of_contrasts_sq) # 10 fordi vi sjekker forskjellene mellom labber, og hver lab har 10 samples.
print(T_0)
crit_t_val = stats.t.ppf(0.975, 60)
print(-1*crit_t_val) # gitt at abs(T_0) > vil H_= bli avvist, og konkludere med at det muligens er forskjell.

# Removing Variables from Model
# If we Remove a variable (Block Variable, Person), we must add the SS from that to the residuals.
New_SS_R = 15413.2 + 502.6
new_df_e = 9+18
SS_tr = 306.1
df_tr = 2
MS_tr = SS_tr/df_tr
MS_er = New_SS_R/new_df_e
F_val = (MS_tr/MS_er)
p_val = stats.f.sf(F_val, df_tr,new_df_e)
# print(p_val) # H_0 is accepted, as p_value is 0.77 # We would need 30 different persons for it to be correct.





