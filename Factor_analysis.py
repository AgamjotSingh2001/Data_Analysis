#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:02:40 2023

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import time

student_data=pd.read_csv("student-mat.csv")
print(student_data.describe())

col_str=student_data.columns[student_data.dtypes==object]
print(col_str)

student_data=pd.get_dummies(student_data,columns=col_str,drop_first=True)
print(student_data.info())

print(student_data[["G1","G2","G3"]].corr())
student_data.drop(axis=1,labels=["G1","G2"])
label=student_data["G3"].values
print(student_data.shape)
predictors=student_data.drop(axis=1,labels=['G3']).values

#Using Linear refression to predict grades
from sklearn.linear_model import LinearRegression
lr=linear_model.LinearRegression()

#Return an array of score of the estimator for each run of the cresso validation
lr_score=cross_val_score(lr, predictors,label,cv=5) #Five runs, 5 means
print("LR Model Class Validaion score : "+str(lr_score))

#Average of all means
print("LR Model Cross Validation mean score :"+ str(lr_score.mean()))

#Using PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=len(student_data.columns)-1)
pca.fit(predictors)
variance_ratio=pca.explained_variance_ratio_
print(pca.explained_variance_.shape)

#Now plot
import matplotlib.pyplot as plt

#Find cumulative variance, adding one independent variable at a time
variance_ratio_cum_sum=np.cumsum(variance_ratio)
print(variance_ratio_cum_sum)

#This cumulative explained variance graph helps us to choose the number of desired principal components. If we look at the above print 
#atatement's output, we will realize that 90% variation in the data is explaining by the first principle components. Hence, we 
#annotate on the graph.

plt.plot(variance_ratio_cum_sum)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

#Annotate 90% variance explained by the first 6 variables only
plt.annotate('6', xy=(6,0.90))

#Individual explained variance instead of cumulative variance
#We see that the first variable cause 60% variance, the speed 22%, and so on ....
plt.Figure(figsize=(10,5))

plt.bar(range(41),pca.explained_variance_,alpha=0.5,label='individual explained variance')
plt.xlabel("Explained variance ratio")
plt.ylabel("Principal camponent")
plt.legend(loc="best")
plt.show()

import seaborn as sns
correlation=pd.DataFrame(predictors).corr()
sns.heatmap(correlation,vmax=1,square=True,cmap="Greens")
plt.title("correlation between different features")
plt.show()

pca=PCA(n_components=6)
pca.fit(predictors)
Transformed_vectors=pca.fit_transform(predictors)
print(Transformed_vectors)

#Correlation between the 6 variables after transforming the data with PCA is 0
correlation=pd.DataFrame(Transformed_vectors).corr()
sns.heatmap(correlation,vmax=1,square=True,cmap="viridis")
plt.title("correlation between different features")
plt.show()

#Check the performance with 6 variables
lr_pca=linear_model.LinearRegression()
lr_pca_score=cross_val_score(lr_pca,Transformed_vectors,label,cv=5)
print("PCA model cross validation score : "+str(lr_pca_score))
print("PCA model cross validation score : "+str(lr_pca_score.mean()))
#We see value similar to the earlier can when  we had 40 independent variable
#This means that PCA has indeed reduced 40 variables to 6 without causing any negative impact

