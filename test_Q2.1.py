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
df_india=df[(df['Country']=="India")]


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
