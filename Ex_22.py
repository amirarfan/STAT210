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

my_data = pd.read_csv("EX14_1Montg_data.csv", sep=";", header=0)
# print(my_data.head())

my_mod = ols("Purity ~ C(Supplier, Sum)/C(Batch, Sum)", data=my_data).fit()
anov = sm.stats.anova_lm(my_mod)
# print(anov)
mu = my_data["Purity"].mean()
sigma_error = anov.mean_sq["Residual"]
sigma_t = (anov.mean_sq["C(Supplier, Sum)"]-anov.mean_sq["C(Supplier, Sum):C(Batch, Sum)"])/(3*4)
sigma_b = (anov.mean_sq["C(Supplier, Sum):C(Batch, Sum)"]-anov.mean_sq["Residual"])/3

# print(round(mu, 2), round(sigma_error,2), round(sigma_t,2), round(sigma_b,2))

# c)
# First for supplier
# H_0: sigma_t = 0 , H_A: sigma_t > 0
F_val = anov.mean_sq["C(Supplier, Sum)"]/anov.mean_sq["C(Supplier, Sum):C(Batch, Sum)"]
p_val = st.f.sf(F_val, 2, 9)
# print(p_val) # Aksepter H_0.

# For Batch nested innen Supplier
# H_0 :  Sigma_b = 0, H_A: Sigma_b > 0
F_val = anov.mean_sq["C(Supplier, Sum):C(Batch, Sum)"]/anov.mean_sq["Residual"]
p_val = st.f.sf(F_val, 9, 24)
# print(p_val) # Avviser H_0

# d) Konfidensintervall
se_mu = math.sqrt(anov.mean_sq["C(Supplier, Sum)"]/3)
df_mu = 3 - 1
crit_val = st.t.ppf(1-0.025, df_mu)
lower = mu - crit_val*se_mu
upper = mu + crit_val*se_mu
# print(lower, upper)


with localconverter(ro.default_converter + pandas2ri.converter):
    R_df = ro.conversion.py2ri(my_data)

col_lake_index = list(R_df.colnames).index("Supplier")
col_lake = ro.vectors.FactorVector(R_df.rx2("Supplier"))
R_df[col_lake_index] = col_lake

hook_lake_index = list(R_df.colnames).index("Batch")
hook_lake = ro.vectors.FactorVector(R_df.rx2("Batch"))
R_df[hook_lake_index] = hook_lake

car = importr("car")
mixlm = importr("mixlm")
stats = importr("stats")
lme_4 = importr("lme4")
base = importr('base')

fml = Formula("Purity ~ r(Supplier) + r(Batch)%in%r(Supplier)")
# mod = mixlm.lm(fml, data=R_df)
# Anov = mixlm.Anova_lmm(mod, data=R_df, type=StrVector("III"))
mod = mixlm.lm(fml, data=R_df, REML=True)
# print(base.summary(mod))
conf_int = lme_4.confint_merMod(mod)
print(conf_int) # Remark that these are for the deviations and not for variance, square them for the variance.