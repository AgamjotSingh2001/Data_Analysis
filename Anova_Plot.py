

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

df=pd.read_csv('StudentsPerformance.csv')

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

'''  Fom our data summary, we can see that there is no null value indicating
 we are workinfg with clean data set. additionally thr score of maths , Reading and writing contain similer averge'''
 
def distribution(dataset,variable):
    '''
    Args:
        dataset=Include the dataframe here
        variable=Include the column from dataframe used for color
    encoding:
        Returns:
            sns pairplot with color encoding
            '''
    g=sns.pairplot(data=dataset,hue=variable)
    g.fig.suptitle('Graph showing distribution between scores and {}'.format(variable),fontsize=20)
    g.fig.subplots_adjust(top=0.9)
    return g

print(df.columns)
distribution(df,'gender')
plt.show()

#Female perform higher in reading and writing whike makes perform higher on math.

#score and race
distribution(df, 'race/ethnicity')
plt.show()

#Score and parental education level
distribution(df, 'parental level of education')
plt.show()
#There appears to be a trend in parental education level and students's score. The variance between the categorical data indicated this is not a major factor

#Score and lunch
distribution(df, 'lunch')
plt.show()
#Students who ate standard lunch on average tested higher in all 3 subjects

distribution(df, 'test preparation course')
plt.show()
#students who completed test preparation course on average tested higher in all 3 subjects

'''finding co relation between categorial variables using test scores using 1-way ANOVA

1- Way ANOVA hypothesis

Null hypothesis (H0): There is a difference between group and quality between means 
Alternative hypothesis (H1): There is a difference between the means and groups.
'''

#Clean up column names for Statas models
df.columns = ['gender', 'race', 'parental_edu', 'lunch', 'test_prep_course', 'math_score', 'reading_score', 'writing_score']

#Create anova test function
def anova_test(data, variable):
    '''
    Args: 
        data = (DataFrame)
        variable = Categorical column used for 1-way ANOVA test
    Returns: Nothing
    '''
    x = ['math_score', 'reading_score', 'writing_score']
    for i,k in enumerate(x):
        lm = ols('{} ~ {}'.format(x[i],variable), data=data).fit()
        table = sm.stats.anova_lm(lm)
        print("P-value for 1-way ANOVA test between {} and {} is ".format(x[i],variable),table.loc[variable,'PR(>F)'])

#Gender anova
anova_test(df, 'gender')

#parental education anova
anova_test(df, 'parental_edu')

anova_test(df, 'lunch')

anova_test(df, 'test_prep_course')

#Students who ate the standard lunch tested higher than those who ate a free/reduced meal.
#Students who completed a test preparation course scored higher than those who did not.

#All categorical data was statistically tested against the exam scores using a 1-Way ANOVA test. This test allows us to accurately confirm whether a category of data is correlated to the numerical outcome. Using a 95% confidence internal we acheived p-values < 0.05 for each catergory of data. This allows us to reject our null hypothesis and summize that the catergorical data in this dataset is correlated to the reading, writing, and math scores.

#Create countplot fpr parental education and student scores 
plt.figure(figsize=(12,5))
sns.countplot(data=df,x='parental_edu',hue='gender')
plt.show()
'''
#Our dataset included a very low number of parents with a master's degree or bachelor's degree. Due to the low samplw size we can not confidentaly say that students with highly educated parents will score better.

Coclusion:
    
Females perform higher iin reading and writing subjects.
Males perform higher in math, 
Parental education level has a negiligible difference in student's test performance.
Students who ate the standard lunch tested higher than those who ate a free/reduced meal.
Students who completed a test preparation course scored higher than those who did not.

All categorical data was statistically tested against the exam scores using a 1-Way ANOVA test. This test allows us to accurately confirm whether a category of data is correlated to the numerical outcome. Using a 95% confidence internal we acheived p-values < 0.05 for each catergory of data. This allows us to reject our null hypothesis and summize that the catergorical data in this dataset is correlated to the reading, writing, and math scores.
'''