import math
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import binom
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import numpy as np

# a)
# The factor store is of no interest, but we have to keep it in the model as it contributes to the explained variance.
# It might improve the test power.


#
my_data = pd.read_csv("ListeriaStore.csv", sep=";", decimal=",", header=0, index_col=0)
my_data["ListeriaNumber"] = np.log(my_data["ListeriaNumber"])
# print(my_data.head())
#c)
my_mod = ols("ListeriaNumber ~ C(HamTopping, Sum) + C(GroceryStore, Sum)", data=my_data).fit()
anov = sm.stats.anova_lm(my_mod)
# print(anov)
# print()
#d) Fitting the model without using store as a block
non_block_mod = ols("ListeriaNumber ~ C(HamTopping, Sum)", data=my_data).fit()
non_block_anov = sm.stats.anova_lm(non_block_mod)
# print(non_block_anov)

# e) As it is controllable one should probably keep it in the model.

# f) If added as a random factor one could interpret the covariance, but it has no effect on the other factors.

fish_data = pd.read_csv("FishingExperimentwPersons.csv", header=0, sep=";", decimal=",")
print(fish_data.head())
fish_mod = ols("Yield ~ C(Lake, Sum)*C(Hook, Sum) + C(Person, Sum)", data=fish_data).fit()
fish_anov = sm.stats.anova_lm(fish_mod)
print(fish_anov)

# Having them as random or fixed does not matter for the ANOVA table, but adding a block affects the error term in terms of SS and df