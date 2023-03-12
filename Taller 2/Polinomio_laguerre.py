import sympy as sym
import numpy as np
from fractions import Fraction
from sympy.solvers.solveset import linsolve

n=3
x=sym.Symbol("x",real=True)
y=sym.Symbol("y",real=True)

def GetLegendre(n,x,y):
    
    y=((x**2)-1)**n
    
    poly=sym.diff(y,x,n)/(2**n*np.math.factorial(n))
    poly=sym.simplify(poly)
    return poly

Legendre=[]

for i in range(n):
    
    Legendre.append(GetLegendre(i,x,y))
    
a, b, c = sym.symbols('a, b, c')
p0, p1, p2 = sym.symbols('p0, p1, p2')



sol,=linsolve([a-c*1/2 -3,b-5,3/2*c -1],(a,b,c))

P=sol[0]*p0,sol[1]*p1,sol[2]*p2

print(P)
    
    