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


# Anova - Fixed Model
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

# Tukey Fixed Effects Model

