#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:08:13 2023

@author: dai
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import probplot
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
shoes=pd.read_csv("women_shoe_prices.csv")
print(shoes.shape)
shoes.sample(5)
shoes["midprices"]=(shoes["prices.amountMax"]+shoes["prices.amountMin"])/2
probplot(shoes["midprices"],dist="norm",plot=pylab)
plt.show()

#The gr
pink=shoes[shoes.colors=="Pink"]
notpink=shoes[shoes.colors!="Pink"]

print(ttest_ind(pink.midprices,notpink.midprices,equal_var=False))

plt.hist(shoes['midprices'])
plt.show()

shoesreduced=shoes[shoes.midprices<300]
entries_removed=shoes.shape[0]-shoesreduced.shape[0]
print("removed shoes that are under 300:%d\n"%entries_removed)
shoes=shoes['midprices']
plt.show()

#plot a qqplot to check normality. If the variable is normally distributed, most of the points should be along the ceter diagonal
probplot(shoes["midprices"],dist="norm",plot=pylab)
plt.show()

#The graph now is closer to the straight line - which is more closer  to a normal distribution curve
pink=shoes[shoes.colors=="Pink"]
notpink=shoes[shoes.colors!="Pink"]


print(ttest_ind(pink.midprices,notpink.midprices,equal_var=False))
#the p-value is still under 0.01 which shows there is significant that pink shoes are different price to other shoes