import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
df=pd.read_csv('data_date.csv')


df['Date_modified']=pd.to_datetime(df['Date'])
print(df.head())
df.set_index(df['Date_modified'],inplace=True)
print(df.head())
df.drop(['Date'], axis=1, inplace=True)
print(df.head())
print(df.shape)

df_cyprus=df[(df['Country']=="Cyprus")]
print(df_cyprus)
X=df_cyprus['Date_modified']
mean=df_cyprus['AQI Value'].mean()
std=df_cyprus['AQI Value'].std()
print(mean)
print(std)
Y=df_cyprus['AQI Value']
plt.figure(figsize=(10,6))
sns.boxplot(Y)
plt.plot()                                