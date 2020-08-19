import math

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from scipy import stats
from scipy.special import binom
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import numpy as np


# H_0 = Contrasts = 0, H_A = Contrasts != 0
my_data = pd.read_csv("SiresData.csv", sep=";", header=0)
my_data["Sire"] = my_data["Sire"].astype("category")
grouped_df = pd.DataFrame(my_data.groupby("Sire")["Milk"].mean(), columns=["Milk"])
grouped_df = grouped_df.reset_index()

mod = ols("Milk ~ C(Sire, Sum)", data=my_data).fit()
# print(mod.summary())
anov = sm.stats.anova_lm(mod)
# print(anov)

# print(grouped_df)

# 11a)
# Definerer kontraster 1/2(mean_1 + mean_4) - 1/3(mean_2 + mean_3 + mean_5)
# H0 : Kontraster = 0, H_A : Kotnraster != 0
# Summerer kontraster med spådde mean verdier

sum_cont = 1/2*(grouped_df["Milk"][0] + grouped_df["Milk"][3]) - 1/3*(grouped_df["Milk"][1] + grouped_df["Milk"][2] + grouped_df["Milk"][4])
# print(sum_cont)
se_val = math.sqrt((anov.mean_sq["Residual"]/8)*(0.5**2 + 0.5**2 + 0.333**2 + 0.333**2 + 0.333**2))
T_val = sum_cont/se_val
crit_t_val = stats.t.ppf(1-0.025, df=35)
# print(T_val)

# print(T_val > crit_t_val)

# True , kan avvise H_0 ved signifkantnivå 0.05.

#11 b)
antall_komb = binom(5, 2)
# print(antall_komb) # eventuelt math.comb, 10 kombinasjoner totalt

# 11 c)
comparisons = MultiComparison(my_data["Milk"], my_data["Sire"])
# print(comparisons.tukeyhsd(alpha=0.01).summary())

# 11 d)
pivot_data = my_data.pivot(columns="Sire", values="Milk")
pivot_data = pivot_data.apply(lambda x: pd.Series(x.dropna().values))
pivot_data[3][3] = np.NaN
pivot_data[3][6] = np.NaN
print(pivot_data)