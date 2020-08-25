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


#b)
# E(MSA) = sigma^2 + 6*sigma^2tb+ f(t)
#  E(MSB) = sigma^2 + 6*sigma^2tb + 24sigma^2b
# E(MSAB) = sigma^2 +6sigma^2tb
# E(MSE) = sigma^2

#c ) Samme som i 17.

fish_data = pd.read_csv("FishingExperiment.csv", sep=";", header=0, decimal=",")
fish_mod = ols("Yield ~ C(Hook, Sum)*C(Lake,Sum)", data=fish_data).fit()
anov = sm.stats.anova_lm(fish_mod)
print(anov)