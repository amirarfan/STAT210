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
print(my_data["SalmonLice"].mean())
print(my_mod.summary())
anov = sm.stats.anova_lm(my_mod)
print(anov)
#
