# R1: labor     :    2*x1   + 1*x2      + 2.5*x3 <= 60 Hours
# R2: Machine   :   0.8*x1  + 0.6*x2    + 1.0*x3 <= 16 Hours
# R3: Wood      :   30*x1   + 20*x2     + 30*x   <= 400 board-feet

#import all classes from PuLP model
from pulp import *

#Create the probelm variable to contain the problem data

x1=LpVariable("tables",0,None,LpInteger)
x2=LpVariable("chairs",0,None,LpInteger)
x3=LpVariable("bookcases",0,None,LpInteger)

model+=40*x1+30*x2+45*x3

model+=2*x1+1*x2+2.5*x3<=60,"Labour"
model+=0.8*x1+0.6*x2+1.0*x3<=16,"Machine"
model+=30*x1+20*x2+30*x3<=400,"wood"

model.solve()

for v in models.variable():
    print(v.name)