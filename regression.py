#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 08:57:45 2023

@author: dai
"""
import pandas as pd
dataset=pd.read_csv('salary.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
print(y_pred)

import matplotlib.pyplot as plt
plt.scatter(X_train, y_train,color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experiance(Training set)')
plt.xlabel('Years of experiance')
plt.ylabel('Salary')
plt.plot()


plt.scatter(X_test, y_test,color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experiance(Test set)')
plt.xlabel('Years of experiance')
plt.ylabel('Salary')
plt.plot()

#Make new prediction

new_salary_pred=regressor.predict([[15]])
print('The Predicted salary iof a person with 15 years experiance is',new_salary_pred)