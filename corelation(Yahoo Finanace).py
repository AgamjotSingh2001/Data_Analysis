import numpy as np
import pandas as pd
import yfinance as yf

tickers=['TCS.NS','ICICIBANK.NS','RELIANCE.NS','BHARTIARTL.NS','ITC.NS','MARUTI.NS','BAJFINANCE.NS']
df=yf.download(tickers,period="3y")
close=df.Close

norm=close.div(close.iloc[0].mul(100))
ret=close.pct_change().dropna()
print(ret)
print(ret.cov())
print(ret.corr())

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,7))
sns.heatmap(ret.corr(),cmap='Blues')
plt.show()

sns.heatmap(ret.corr(),cmap='Blues',annot=True)
plt.show()

sns.heatmap(ret.cov(),cmap='Blues',annot=True)
plt.show()