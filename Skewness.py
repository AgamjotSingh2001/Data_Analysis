import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(33)
norm_sample=np.random.randn(1000)
sns.histplot(x=norm_sample,kde=True)
plt.show()

print('mean=',np.mean(norm_sample))
print('median=',np.median(norm_sample))
print('std=',np.std(norm_sample))

#Generate a sample of 10,000 data points from a right skewed distribution
#Firat parameter if 0 means no skewness, +ve means positive skewness,-ve means negative skewness
#This means the generated data will haave a +ve skew, i.e. long tail on the right side, and the majority of the data will be concentrated on the left side.
#10 here is just 10 units, ig you increase it, the right skewness will inrease even further
skewed_data_R=stats.skewnorm.rvs(10,size=10000)
sns.histplot(x=skewed_data_R,kde=True)
plt.show()

print("Mean=",np.mean(skewed_data_R))
print("Median=",np.median(skewed_data_R))

#Generate a sample of 10,000 data points from a left-skewed distribution

skewed_data_L=stats.skewnorm.rvs(-10,size=10000)
sns.histplot(x=skewed_data_L,kde=True)
plt.show()

print("Mean=",np.mean(skewed_data_L))
print("Median=",np.median(skewed_data_L))

#Q-Q plot
stats.probplot(norm_sample,plot=plt)
plt.show()
stats.probplot(skewed_data_R,plot=plt)
plt.show()
stats.probplot(skewed_data_L,plot=plt)
plt.show()

#print skewness of each data
print(stats.skew(norm_sample))
print(stats.skew(skewed_data_R))
print(stats.skew(skewed_data_L))

#perform a skewness test on each dataset
#returns a numerical value representing the skewness of the dataset
print(stats.skewtest(norm_sample))
print(stats.skewtest(skewed_data_R))
print(stats.skewtest(skewed_data_L))

#Output : statistic: The Statistic value is the test statistic calculated by the skewness test. It quantifies  the degree of skewness in the dataset.
#if negative, it indicated that the dataset is  left-skewed or negatively skewed

#The p-value neasures the evidence against ta null hypothsis
#A small p-value (typiccally less than 0.05) suggest that there is strong evidence to reject the null hypothesis, indicating that the data is significantly skewed.
#A larger p-value (greater tahn 0.05) suggest that there is not enough evidence to reject the null hypothsis, indicating that the data may bit be significantly skewed.