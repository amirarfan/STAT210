from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects import Formula
from rpy2.robjects.vectors import StrVector
import math
import seaborn as sns
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

#


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

fml = Formula("Yield ~ r(Lake)*r(Hook)")
mod = mixlm.lm(fml, data=R_df)
Anov = mixlm.Anova_lmm(mod, data=R_df, type=StrVector("III"))
mod = mixlm.lm(fml, data=R_df)
print(R.summary(mod))
