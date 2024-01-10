import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

housing=pd.read_csv("housing.csv")
print(housing.head())
print(stats.skewtest(housing))
sns.histplot(data=housing,x='SalePrice',kde=True)
plt.show()


credit=pd.read_csv("Credit.csv")
print(credit.head())

sns.histplot(data=credit,x='Education',kde=True)
plt.show()


insurance=pd.read_csv("insurance(1).csv")
print(insurance.head())

sns.histplot(data=insurance,x='bmi',kde=True)
plt.show()



stocks=pd.read_csv("stocks.csv")
print(stocks.head())

sns.histplot(data=stocks,x='apple',kde=True,bins=30)
plt.show()