#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:14:23 2023

@author: dai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print(sns.get_dataset_names())
df=sns.load_dataset('flights')
print(df.head())

df['yearMonth']="01-"+df['month'].astype(str)+"-"+df['year'].astype(str)
print(df.head())

#Make yearMonth column as the dataframe index
df['yearMonth']=pd.to_datetime("01-"+df['month'].astype(str)+"-"+df['year'].astype(str))
print(df.info())
print(df.head())

df.set_index('yearMonth',inplace=True)
print(df.head)

"""plt.figure(figsize=(10,5))
sns.lineplot(data=df,x=df.index,y=df.passengers)
plt.show()

#Calculate the plot rolling mean and standard deviation for 12 month
df['rollMean']=df.passengers.rolling(window=12).mean()
df['rollStd']=df.passengers.rolling(window=12).std()

print(df['rollMean'])
print(df['rollStd'])

plt.figure(figsize=(10,5))
sns.lineplot(data=df,x=df.index,y=df.passengers,label="passengers")
sns.lineplot(data=df,x=df.index,y=df.rollMean,label="rolling Mean")
sns.lineplot(data=df,x=df.index,y=df.rollStd,label="rolling standard deviation")
plt.legend();
plt.plot()

from statsmodels.tsa.stattools import adfuller
adfTest=adfuller(df['passengers'])
print(adfTest)

stats=pd.Series(adfTest[0:4],index=['test Statistic','p-value','#lags used','number of observation used'])
print(stats)
for key,values in adfTest[4].items():
    print("citicality",key,":",values)

def test_stationarity(dataFrame, var):
    dataFrame['rollMean']=df.passengers.rolling(window=12).mean()
    dataFrame['rollStd']=df.passengers.rolling(window=12).std()
    
    from statsmodels.tsa.stattools import adfuller
    adfTest=adfuller(dataFrame[var])
    stats=pd.Series(adfTest[0:4],index=['test Statistic','p-value','#lags used','number of observation used'])
    print(stats)
    for key,values in adfTest[4].items():
        print("citicality",key,":",values)
    sns.lineplot(data=dataFrame,x=dataFrame.index,y=dataFrame.passengers,label="passengers")
    sns.lineplot(data=dataFrame,x=dataFrame.index,y=dataFrame.rollMean,label="rolling Mean")
    sns.lineplot(data=dataFrame,x=dataFrame.index,y=dataFrame.rollStd,label="rolling standard deviation")
    plt.legend();
    plt.plot()
        
#Just get the passageners column into a new dataframe for easier testing 
air_df=df[['passengers']].copy() #Double brackets because it is a list within a list
print(air_df.head())

#By default, shift is by 1 time period (here, one month)
#create a new column which will contain the shifted value from passangers columns - see slide
air_df['shift']=air_df.passengers.shift(10)
air_df['shiftDiff'] = air_df['passengers'] - air_df['shift']
print(air_df.head())

#test staionarity
test_stationarity(air_df.dropna(), 'shiftDiff')
#Conclusion : The data has become somewhat stationary
"""
#Create column for one month and one year lagged data
airP=df[['passengers']].copy(deep=True)
airP['firstDiff']=airP['passengers'].diff()
airP['Diff12']=airP['passengers'].diff(12)


print(airP.head())


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


plot_pacf(airP['firstDiff'].dropna(),lags=20)
plt.show()

plot_acf(airP['firstDiff'].dropna(),lags=20)




