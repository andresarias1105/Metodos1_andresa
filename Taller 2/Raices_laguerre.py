import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

##esto casi funciona xd

# Definimos los símbolos que usaremos
x = sym.Symbol('x')
y = sym.Symbol('y')
n = 20

# Creamos una lista para almacenar todas las raíces
all_roots = []
def GetLaguerre(n,x,y):
    
    y=(x**n)*(sym.exp(-x))
    
    poly=sym.diff(y,x,n)/(np.math.factorial(n))
    
    return poly*(sym.exp(x))

Laguerre=[]
DLaguerre=[]



for j in range(1,n+1):
    Poly=GetLaguerre(j,x,y)
    Poly=sym.simplify(Poly)
    DPoly=sym.diff(Poly,x,1)
    Laguerre.append(Poly)
    DLaguerre.append(DPoly)
    
def GetNewtonRhapson(f,df,xn,itmax=100,precision=1e-14):
    
    error=1
    it=0
    
   
    
        
    while error > precision and it<=itmax:
        
          try:
            xn1=xn-f(xn)/df(xn)
            
            error=np.abs(f(xn)/df(xn))
          except ZeroDivisionError:
            print("division por cero")            
         
          it+=1
        
        
          xn=xn1
          
    if it == itmax:
        False
    else:
        return xn
    
    

    
    
    return xn

def GetAllRoots(n, xn, tolerancia=5):
    Roots = np.array([])
    
    func = sym.lambdify([x], Laguerre[n], 'numpy')
    dfunc = sym.lambdify([x], DLaguerre[n], 'numpy')
    Roots = np.array([])
    
    for i in xn:
        
        root = GetNewtonRhapson(func,dfunc,i)
        
        if root != False:
            
            croot = np.round( root, tolerancia )
            
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots


xn = np.linspace(0,100,200)

for i in range(1,n):

  Roots = GetAllRoots(i,xn)
  

  
  print((Roots))

