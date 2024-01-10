import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("loan.csv")
correlation_coefficient=df['income'].corr(df['age'])
print(f'correlation coefficent:{correlation_coefficient}')
plt.scatter(df["income"], df["age"])
plt.xlabel('income')
plt.ylabel('age')
plt.title(f'scatter plot(correlation:{correlation_coefficient:.2f})')

#Identify potential outliers (e.r., value with residuals greater than 2 times the standard deviation)
residuals = df['age']- df['income']
print(residuals)

std_deviation=residuals.std()
outliers = df[abs(residuals)>2*std_deviation]

#Highlight potential outliers on the scatter plot
plt.scatter(outliers['age'], outliers['income'], color='red',label='outliers')
plt.legend()
plt.show()