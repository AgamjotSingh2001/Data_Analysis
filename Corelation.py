from sklearn.datasets import fetch_california_housing

'''
return_x_y=True: This parameter specifies that you want the dataset to be split into features (in this case, housing) and target
values (in this case, target) and returned as separated variables. When return_X_y is set to True, housing will contain the feature data, and 
target will contain the target values.
The feature variables and target values are already decided for this dataset.
please refer to this: https:scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
'''

housing, target =fetch_california_housing(as_frame=True, return_X_y=True)
print(housing.head())
print(target.head())
print(housing.corr())
corr=housing.corr()
print(corr['MedInc']["AveRooms"])
#print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt
cmap=sns.diverging_palette(10,220,as_cmap=True)
sns.heatmap(corr,vmin=-1.0,vmax=1.0,square=True,cmap=cmap)
plt.plot()