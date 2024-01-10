import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
df=pd.read_csv('data_date.csv')

print(df.head())
print(df.describe())

'''
Question 1: Apply Hypothesis test to check if AQI of July and August months, all over the world are significantly different then all
around the world.

H0 (Null hypothesis) : (It implies that the mean of all the popluations are equal)
H1 (Alternate Hypothesis) : It states that there will be at least one population mean that differs from the rest
'''


df['Date_modified']=pd.to_datetime(df['Date'])
print(df.head())
df.set_index(df['Date_modified'],inplace=True)
print(df.head())
df.drop(['Date'], axis=1, inplace=True)
print(df.head())
print(df.shape)

new_df=df[(df['Date_modified'].dt.month==7) | (df['Date_modified'].dt.month==8)]
print(new_df.shape)

result=new_df.groupby('Country')['AQI Value'].apply(list)
print(result)

from scipy.stats import f_oneway
F,p=f_oneway(*result)
print(F)
print("p VALUE:",p)

#CONCLUSION :P=0.sINCE THE P_VALUE=0 WHICH IS LESS THAN 0.05,WE REJECT THE NULL HYPOTHESIS, SO THERE I DIFFERENCE BETWEEN THE MEAN aQI VALUES OF DIFFERENT COUNTRY

"""2 DRAW VISUALIZATION
1. DAY WISE AQI of India """

df_india=df[(df['Country']=="India")]
df_india.drop(['Date_modified'], axis=1, inplace=True)

print(df_india)
X=df_india["Date_modified"]
Y=df_india["AQI Value"]
print(X)
print(Y)
plt.figure(figsize=(15,8))
plt.bar(X,Y)
plt.xlabel("Dates")
plt.ylabel("AQI")
plt.plot()
