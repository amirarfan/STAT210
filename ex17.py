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
# from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from scipy.special import binom
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import numpy as np
import pprint as pp

utils = importr('utils')
R = ro.r



my_df = pd.read_csv("FishingExperiment.csv", sep=";", decimal=",", header=0)

# 17  c)
# Hypothesis test:
# H_0 : sigma_tb^2 = 0 , H_A: sigma_tb^2 > 0
# f_val = 33.255/2.578
# crit_f_val = st.f.ppf(0.95, 3, 40)
# print(f_val)
# print(crit_f_val)
# p_val = st.f.sf(f_val, 3, 40)
# print(f_val > crit_f_val)
# print(p_val) # Reject null hypothesis

# H_0 : sigma_t^2 = 0, H_A: sigma_t^2 > 0
# f_val = 41.865/33.255
# crit_f_val = st.f.ppf(0.95, 3, 3)
# p_val = st.f.sf(f_val, 3, 3)
# print(p_val) # Accept the Nullhypothesis

# #H_0: sigma_b^2 = 0, H_A: sigma_b^2 > 0
# f_val = 79.053/33.255
# crit_f_val = st.f.ppf(0.95, 3, 3)
# p_val = st.f.sf(f_val, 3, 3)
# print(p_val) # Accept Null Hypothesis

# d) mu = 9.93 sigma_t_sq = (41.865 - 33.255) / (2*6) etc ... Out of the
# results one can say that the variance components for the treatment effects
# alone are not very significant implying that alone they are not very
# relevant, but the interaction term shows significant variance.
# meaning that one should use different combinations of lake and hook
# to achieve the best results.




with localconverter(ro.default_converter + pandas2ri.converter):
    R_df = ro.conversion.py2ri(my_df)

col_lake_index = list(R_df.colnames).index("Lake")
col_lake = ro.vectors.FactorVector(R_df.rx2("Lake"))
R_df[col_lake_index] = col_lake

hook_lake_index = list(R_df.colnames).index("Hook")
hook_lake = ro.vectors.FactorVector(R_df.rx2("Hook"))
R_df[hook_lake_index] = hook_lake

car = importr("car")
mixlm = importr("mixlm")
stats = importr("stats")
lme_4 = importr("lme4")

fml = Formula("Yield ~ r(Lake)*r(Hook)")
mod = mixlm.lm(fml, data=R_df)
Anov = mixlm.Anova_lmm(mod, data=R_df, type=StrVector("III"))
mod = mixlm.lm(fml, data=R_df, REML=True)
conf_int = lme_4.confint_merMod(mod, level=0.95)
print(conf_int)

