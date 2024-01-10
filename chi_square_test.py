import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

'''
Here we use stats.chisquare
primarily used for conducting a chi-square goodness-of-fit test, which compares 
observed frequencies with expected frequencies under a specified theoretical distribution '''

#Fake demographic data for U.S. and Minnesota state and walk through the Chi-square goodness of the to check whther that are different.
    
national = pd.DataFrame(["white"]*100000+["hispanic"]*60000+["black"]*50000+["asian"]*15000+["other"]*35000)
minnesota=pd.DataFrame(["white"]*600+["hispanic"]*300+["black"]*250+["asian"]*75+["other"]*150)
print(national.sample(5))
print(minnesota.sample(5))

national_table=pd.crosstab(index=national[0], columns="count")
minnesota_table=pd.crosstab(index=minnesota[0], columns="count")
print(national_table)
print(minnesota_table)

observed=minnesota_table
national_ratios=national_table/len(national)
expected=national_ratios*len(minnesota) #Get population ratios

#Calculate the expected counts by multiplying the population ratio in the national dataset by thre total number of observaton in the minnesota dataset.
expected = national_ratios=len(minnesota) #Get expected counts

#Calculate the chi-squared statistic by comaring the observed and expected counts and summing the squared diffr=erences, normalize by the expected counts.
chi_squared_stat=(((observed-expected)**2)/expected).sum()

print("Calculated observed value")
print(chi_squared_stat)

#find the critical value for a 95% confidence level with 4 degrees of freddom.

crit = stats.chi2.ppf(q=0.95, #Find the critical value for 95% confidence*
                      df=4) #Df=number of variable categories -1

print("Critcal value")
print(crit)

#Calculate the p-value based on the chi-squared statistic and degrees of freddom.
p_value = 1 - stats.chi2.cdf(x=chi_squared_stat, df=4)
print("p value=")
print(p_value)

print('OBSERVED VALUE THROUGH STTS LIBRARY')
print(stats.chisquare(f_obs=observed,f_exp=expected))