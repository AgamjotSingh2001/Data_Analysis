import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency

df=pd.read_csv("heart_failure_clinical_records_dataset.csv")
print(df.head())
print("Against Age")
print(chi2_contingency(pd.crosstab(index=df["age"], columns=df["DEATH_EVENT"])))

print("Against Anemia")
print(chi2_contingency(pd.crosstab(index=df["anaemia"], columns=df["DEATH_EVENT"])))

print("Against High blood pressure")
print(chi2_contingency(pd.crosstab(index=df["high_blood_pressure"], columns=df["DEATH_EVENT"])))

print("Against Sex")
print(chi2_contingency(pd.crosstab(index=df["sex"], columns=df["DEATH_EVENT"])))

plt.figure(figsize=(20,6))
title=plt.title("survival and deaths by age", fontsize=20)
title.set_position([0.5,1.15])
ax=sns.countplot(x="age",hue="DEATH_EVENT",data=df)
ax.set_xlabel('Age')
ax.set_ylabel('Count')
a=ax.set_xticklabels(ax.get_xticklabels(),rotation=0,horizontalalignment='center')
plt.show()