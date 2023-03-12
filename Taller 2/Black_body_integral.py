import numpy as np
import matplotlib.pyplot as plt

import sympy as sym

def GetNewton(f,df,xn,itmax=1000,precision=1e-12):
    
    error = 1.
    it = 0
    
    while error >= precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/df(xn)
            
            error = np.abs(f(xn)/df(xn))
            
        except ZeroDivisionError:
            print('Zero Division')
            
        xn = xn1
        it += 1
        
    if it == itmax:
        return False
    else:
        return xn
def GetRoots(f,df,x,tolerancia = 10):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)
        
        if root != False:
            
            croot = np.round( root, tolerancia )
            
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots


n=10#orden polinomial

##cuadratura 
Roots1,Weights1=np.polynomial.laguerre.laggauss(n)


x=sym.Symbol("x",real=True)
y=sym.Symbol("y",real=True)

def GetLaguerre(n,x,y):
    
    y=(x**n)*(sym.exp(-x))
    
    poly=sym.diff(y,x,n)/(np.math.factorial(n))
    
    return poly*(sym.exp(x))

Laguerre=[]
DLaguerre=[]



for i in range(n+2):
    Poly=GetLaguerre(i,x,y)
    Laguerre.append(Poly)
    DLaguerre.append(sym.diff(Poly,x,1))





def GetAllRoots(grado,xn,Laguerre,DLaguerre):
    
    poly = sym.lambdify([x],Laguerre[grado],'numpy')
    Dpoly = sym.lambdify([x],DLaguerre[grado],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)
    
    return Roots
# Scan

xn = np.linspace(0,100,100)


def GetWeights(Roots,Laguerre,grado):
    
    Poly = sym.lambdify([x],Laguerre[grado+1],'numpy')
    Weights= Roots/(((grado+1)**2)*Poly(Roots)**2)
    
    return Weights
"""
Weights = GetWeights(Roots,Laguerre)
"""

a=0
b=100
f=lambda x:(x**3)/(np.exp(x)-1)



def integrate_Laguerre(f,a,b,grado):
    
    Rootsi=GetAllRoots(grado,xn,Laguerre,DLaguerre)
    
  
    Weightsi=GetWeights(Rootsi,Laguerre,grado)
    
    
    t=Rootsi+a

    integral=np.sum(Weightsi*np.exp(t)*f(t))
    
    return integral


Integral_n_3=integrate_Laguerre(f,a,b,3)

print("La aproximacion con n=3 es:",Integral_n_3)
I=(np.pi**4)/15
error=np.array([])
n_=np.arange(2,n+1)
for i in range(2,n+1):
    integral=integrate_Laguerre(f,a,b,i)
    
    error=np.append(error,integral/I)
    

plt.axhline(y=1,color="k",linestyle="--")
plt.grid(visible=True)
plt.scatter(n_,error,color="b",label="Precision de cuadratura de Laguerre")
plt.legend()




