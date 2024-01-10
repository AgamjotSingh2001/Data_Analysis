from numpy import linalg as la
from sympy.abc import x,y
from sympy import *
from sympy import diff, pprint
import numpy as np

"""h=np.array([[19.6, -1.4],
          [-1.4, 13.6]])
f=np.array([[10.6],
          [6]])
i=np.array([[1],
          [1]])
x=i-np.matmul(la.inv(h),f)
print(x)"""



expr=(x**4)+((0.8)*y**4)+(4*x**2)+(2*y**2)-(x*y)-(0.2*x**2*y)
dx=diff(expr,x)
point=[(x,1),(y,1)]
dx_k=dx.subs(point)
dy=diff(expr,y)
dxx=diff(dx,x)
dxy=diff(dx,y)
dyy=diff(dy,y)
dyx=diff(dy,x)
hessian=np.array([[dxx,dxy],[dyx,dyy]])
mat=np.array([[x],[y]])
for i in range(10):
    dx_k=dx.subs(point)
    dy_k=dy.subs(point)
    dxx_k=dxx.subs(point)
    dxy_k=dxy.subs(point)
    dyy_k=dyy.subs(point)
    dyx_k=dyx.subs(point)
    hessian=np.array([[dxx_k,dxy_k],[dyx_k,dyy_k]])
    hessian_inv=la.inv(hessian)
    f=np.array([[dx_k],[dy_k]])
    mat=mat-np.matmul(hessian,f)
    
print(mat)