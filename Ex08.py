##

import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.formula.api import ols


## Fra Ex08 finner vi ut at SSe = 1.45440

ss_e = 1.4544
conf_inter = ss_e/chi2.isf(q=(0.025, 0.975), df=18)
print(conf_inter)


# c)
# H_0 : sigma^2 = 0.05
# H_A : sigma^2 > 0.05
# Can avvise H_0 hvis SSe/0.05 > X^2_alpha_18 (alpha er signifikantnivået)
chi_sq_vals = chi2.isf(q=(0.01, 0.05, 0.1), df=18)
ss_e_div_chi = 1.4544/0.05

print(ss_e_div_chi > chi_sq_vals)

# konkluderer med at H_0 avvises kun for signifikantnivå 0.01
