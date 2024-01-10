import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

#Sample dataset
'''
Data dictionary defines a time series dataset with three components (Contains three key-value pairs):
'month': The time index or timespan
'Sales': The variable we want to forecast
'Adevertisement': An external variable that may impact the 'Sales' variable, used in the SARIMAX model
'''
data = {
        'Month':pd.date_range(start='2019-01-01',periods=36,freq='M'),
        'Sales':[100, 120, 130, 150, 200, 220, 250, 230, 210, 180, 160, 140, 110, 130, 150, 190, 220, 240, 260, 250, 230, 200, 180, 160, 130, 140, 160, 190, 210, 240, 270, 250, 230, 200, 210, 190],
        'AdvertisingSpend':[50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 70, 120, 90]

}
df=pd.DataFrame(data)
df.set_index('Month',inplace=True)
order=(2,1,1)
model=ARIMA(df['Sales'],order=order)
results=model.fit()

forecast_step=12
forecast=results.forecast(steps=forecast_step)

plt.figure(figsize=(12,6))
plt.plot(df['Sales'],label="Actual result")
plt.plot(pd.date_range(start=df.index[-1],periods=forecast_step,freq="M"),forecast,label='ARIMA forecast',color="red")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("ARIMA forecast for sales")
plt.legend();
plt.show()

# Fit a SARIMAX model to the Sales data with AdvertisingSpend as an exogenous variable 
from statsmodels.tsa.statespace.sarimax import SARIMAX
#SARIMAX allows us to incorporate additional external factors (exogenous variables) that can influence the time series we are trying to forecast. In this case, we are using the 'AdvertisingSpend variable as an exogenous variable. If this variable has a significant impact on 'Sales, including it in the model can improve the accuracy of the forecast.
#SARIMAX includes seasonal components (P, D, Q, s) that can capture seasonality in the data. If our data exhibits seasonal patterns (e.g., sales are higher during specific months of the year), SARIMAX can model and forecast these patterns more effectively.
#Don't use Mobile SARIMAX allows us to specify both non-seasonal (p, d, q) and seasonal (P, D, Q, s) orders. This flexibility enables us to tailor the model to the specific characteristics of our data. In the provided code, we have chosen an order of (2, 1, 1) and seasonal order of (1, 1, 1, 12), which may better capture the data's behavior compared to the ARIMA order

order = (2,1,1) #(p,d,q) order
seasonal_order=(1,1,1,12) #(P,D,Q,S) Seasonal order
exog=df['AdvertisingSpend']

model=SARIMAX(df['Sales'],exog=exog, order=order, seasonal_order=seasonal_order)
results=model.fit()

#Forecast future Sales Values
forecast_steps=24
forecast=results.get_forecast(steps=forecast_step, exog=df.iloc[-forecast_step:]['AdvertisingSpend'])

#Plot the original sales data, AdvertisingSped, and the SAR
plt.figure(figsize=(12,6))
plt.plot(df['Sales'],label='Actual Sales')
plt.plot(pd.date_range(start=df.index[-1],periods=forecast_steps,freq='M'),forecast.predicted_mean,label='SARIMAX Forecast',color='green')
#plt.plot(pd.date_range(start=df.index[-1],periods=forecast_steps, freq='M'),forecast.predicted_mean, label="SARIMAX Forecast", color='green')
plt.plot(df.index[-forecast_step:], df['AdvertisingSpend'][-forecast_steps:],label='AdvertisingSpend',linestyle='--',color='red')    
plt.xlabel('Month')
plt.ylabel('Sales/Advertising Spend')
plt.legend()
plt.show()
    
