import numpy as np
import pandas as pd
from sklearn import tree

input_file="PastHires.csv"
df=pd.read_csv(input_file,header=0)

print(df.head())

#Scikit learn need everything 

d={'Y':1,'N':0}
df['Hired']=df['Hired'].map(d)
df['Interned']=df['Interned'].map(d)
df['Top-tier school']=df['Top-tier school'].map(d)
df['Employed?']=df['Employed?'].map(d)
d2={'BS':0,'MS':1,'Phd':2}
df['Level of Education']=df['Level of Education'].map(d2)
print(df.head())

features=list(df.columns[:6])
print(features)

y=df['Hired']
X=df[features]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X, y)

from IPython.display import Image
from six import StringIO
import pydotplus


#The ecport_graphviz function converts the decision tree classifier into a dot file, and pydotplus converts this dpt file to png or 
#displayable form om jupyter

dot_data=StringIO()
tree.export_graphviz(clf, out_file=dot_data,feature_names=features)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png('job.png')