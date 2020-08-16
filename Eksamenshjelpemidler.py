import numpy as np
import scipy.stats as stats
import math



### Z-Score
# Finne P-verdi fra Z-score
# Sett inn Z score istedenfor tall
# N-setter du til å være 2 hvis two ailed, og 1 hvis one tailed.
# P(X>Z) = 1 tailed, P(X = Z) = Two tailed
z_score = 3.95257
n = 1
prob = (n*stats.norm.pdf(z_score))
print(prob)

# For P(X<Z) Altså alle mindre enn pam
prob = (stats.norm.cdf(z_score))

# Finne Z-score fra P-verdi cutoff
Zskaar = stats.norm.ppf(0.03)

# Finne Kritisk Z-verdi for 95%
z_star = stats.norm.ppf(0.975)

conf_int = stats.norm.interval(0.95, loc=0.25, scale=0.01369)
#-------------------------

### T-Verdi
#Regne ut T-Verdi for lineær regresjon
b_1 = 0.7225
b_0 = 0
SE = 0.0706
T = (b_1 - b_0)/SE

## Finne Kritisk t-verdi for 95%
df = 21
print(stats.t.ppf(0.975,df))

# Finne kritisk t-verdi for 98%
print(stats.t.ppf(0.99,df))

# Finne P-verdi for t
T = 2.75514
df = 44
print(stats.t.sf(T, df)*2) # Ganges med 2 siden det er to-sidig test

#Konfidens intervall for T-intervall lineær regresjon
df = 44 # Antall frihetsgrader
x_9 = 3.5332 # Stigningtallet for andel ikke hvite
SE = 1.2824  # Standardavviket for andel ikke hvite
t_crit = stats.t.ppf(0.975, df) # Finner den kritiske verdien for 95% konfidens intervall
conf_int = x_9 - t_crit*SE, x_9 + t_crit*SE # Beregner konfidensintervallet
print(conf_int)

# Konfidens intervall for T-intervall (Kan også brukes til Multippel Regresjon)
df_1 = 22-1
df_2 = 22-1
df = min(df_1,df_2)
mean_1 = 4.9
mean_2 = 6.1
mean = mean_1 - mean_2
SE  = math.sqrt((1.8**2)/22 + (1.8**2)/22)

stats.t.interval(0.95,df,loc=mean,scale=SE)

# -----------------------
### Sannsynlighet

##Binomisk Sannsynlighet
k = 3
n = 6
p = 1/6
prob = stats.binom.pmf(k, n, p)
print(prob)

# Binomisk Cumulative (P(X<x)) # I dette tilfellet mindre enn k
k = 2
prob = stats.binom.cdf(k,n,p)
#Binomisk SF 1-cdf P(X>x)
prob = stats.binom.sf(k,n,p)

# Binomisk mellom to variabler P(x1 <X < x2)
p = 0.2
n = 20
x2 = 5
x1 = 3-1
# P(x<=5)
prob_2 = stats.binom.cdf(x2,n,p)
#P(3<=x)
prob_1 = stats.binom.cdf(x1,n,p)
prob = abs(prob_2 - prob_1)
print(prob)

# Regne ut forventet vedi
binom = stats.binom(20,0.2)
print(binom.mean())
# Regne ut variansen
print(binom.var())
## Hypergeometrisk
#P(X = 2)
M = 10 # 10 KULER
n = 4 # 4 BLÅ KULER
N = 5 # 5 TILFELDIGE UTVALG
k = 2 # Antall vi vil ha
prob = stats.hypergeom.pmf(k, M, n, N)
print(prob)

#Kumulativ P(X<3)
prob = stats.hypergeom.cdf(3, M, n, N)

#Oppgave med Hypergeometrisk fordeling
M = 50
n = 10
N = 5
k = 2 # P(X<=2)
X = stats.hypergeom(M,n,N)
prob =  X.cdf(k)
Andel_godkjent = prob*N
print(Andel_godkjent)
## Poisson
#Kumulativ
# Sannsynlighet mindre enn 24
prob = stats.poisson.cdf(24,20)
print(prob)

# -----------------------
### ANOVA
# Hvordan bruke Anova i Python
import statsmodels.formula.api as smf
import statsmodels.api as sm
mod = smf.ols('var1 ~ var2', data=vilkaarlig).fit()
anov_tab = sm.anova_lm(mod, typ=2)
print(anov_tab)


# -----------------------
### Forskjell mellom to andeler

def two_proprotions_test(success_a, size_a, success_b, size_b):
    """
    A/B test for two proportions;
    given a success a trial size of group A and B compute
    its zscore and pvalue

    Parameters
    ----------
    success_a, success_b : int
        Number of successes in each group

    size_a, size_b : int
        Size, or number of observations in each group

    Returns
    -------
    zscore : float
        test statistic for the two proportion z-test

    pvalue : float
        p-value for the two proportion z-test
    """
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    prop_pooled = (success_a + success_b) / (size_a + size_b)
    var = prop_pooled * (1 - prop_pooled) * (1 / size_a + 1 / size_b)
    zscore = np.abs(prop_b - prop_a) / np.sqrt(var)
    one_side = 1 - stats.norm(loc=0, scale=1).cdf(zscore)
    pvalue = one_side * 2
    return zscore, pvalue

success_vitaminer = 143
success_ingen_vitaminer = 111
size_vitaminer = 302
size_ingen_vitaminer = 181

z, p = two_proprotions_test(success_vitaminer, size_vitaminer, success_ingen_vitaminer, size_ingen_vitaminer)
# Great fucking shit

syk_preorg = 187
syk_etorg = 135
size_pre_org = 2021
size_et_org = 1919
z,p = two_proprotions_test(syk_preorg, size_pre_org, syk_etorg, size_et_org)
