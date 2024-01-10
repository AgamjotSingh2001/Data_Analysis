import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('diabetes.csv')
print(df)

X=df.drop("Outcome",axis=1)
y=df['Outcome']
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(solver='liblinear')
classifier.fit(X_train,y_train)
y_test_prediction=classifier.predict(X_test)
print(y_test_prediction)

comparision=pd.DataFrame({'Actual':y_test,'Predicted':y_test_prediction})
print(comparision[0:10])

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_test_prediction))

from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_test, y_test_prediction)
print(conf_mat)

plt.figure(figsize=(12,6))
sns.heatmap(conf_mat,annot=True,fmt='d')
plt.title("confusion matrix of test data")
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_prediction))

TN=conf_mat[0][0]
FP=conf_mat[0][1]
FN=conf_mat[1][0]
TP=conf_mat[1][1]

recall=TP/(TP+FN)
print("Recall = ",recall)
precision=TP/(TP+FP)
print("Precision = ",precision)
specificity=TN/(TN+FP)
print("Specificity = ",specificity)
accuracy=(TP+TN)/(TP+FN+FP+TN)
print("Accuracy =",accuracy)






