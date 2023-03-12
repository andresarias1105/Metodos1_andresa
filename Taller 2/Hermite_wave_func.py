import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sympy as sym

deg=20
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


x=sym.Symbol("x",real=True)
y=sym.Symbol("x",real=True)
    
def GetHermite(x,y,n):
    
    y=sym.exp(-(x**2))
    
    Poly=sym.exp(x**2)*sym.diff(y,x,n)
    Poly=(-1)**n * Poly
    
    return sym.simplify(Poly)

Hermite=[]
DHermite=[]

for i in range(0,deg+1):
    
    Poly=GetHermite(x,y,i)
    DPoly=sym.diff(Poly,x,1)
    Hermite.append(Poly)
    DHermite.append(DPoly)
    
def GetAllRoots(grado,xn,Hermite,DHermite):
    
    poly = sym.lambdify([x],Hermite[grado],'numpy')
    Dpoly = sym.lambdify([x],DHermite[grado],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)
    
    return Roots



def Get_Weights(xi,n):
    
    Hn= sym.lambdify([x],Hermite[n-1],'numpy')
    
    num=2**(n-1)*np.math.factorial(n)*np.sqrt(np.pi)
    den=n**2*Hn(xi)**2
    
    return num/den
xn=np.linspace(-10,10,100)
Numerical_Roots=(GetAllRoots(deg,xn,Hermite,DHermite))
Numerical_Weights=Get_Weights(Numerical_Roots,deg)
Np_Roots,Np_Weights = np.polynomial.hermite.hermgauss(deg)

print("Raices-Pesos")
for i in range(len(Numerical_Roots)):
    print(Numerical_Roots[i],Numerical_Weights[i])
    
plt.scatter(Numerical_Roots,Numerical_Weights)
plt.xlabel("Raices")
plt.ylabel("Pesos")
plt.show()
def integrate_Hermite(f,a,b,n,Rootsi,Weightsi):
    
    
 
    
    
    t=Rootsi

    integral=np.sum(Weightsi*f(t,n)*np.exp(t**2))
    
    return integral


def function(xn,n):
   
    Hn= sym.lambdify([x],Hermite[n],'numpy')
    
    psi=(np.pi**(-1/4)*np.exp(-xn**2/2)*Hn(xn))/(np.sqrt(2**n * np.math.factorial(n)))
    
    
    
    return np.abs(psi)**2 *xn**2

a=-100000
b=100000

Integral=integrate_Hermite(function,a,b,1,Numerical_Roots,Numerical_Weights)

print("La Aproximacion de la integral mediante Gauss-Hermite es",Integral)
    

