import sympy as sym 
import numpy as np
import matplotlib.pyplot as plt


##A

def sgn(x):
    
    if x<0:
        return -1
    if x>0:
        return 1
    else:
        return 0
    
##B
func=np.vectorize(sgn)
X0=np.linspace(-1,0,50)
x=np.linspace(0,1,50)

X=np.append(X0,x)
Y=func(X)


##C
Roots,Weights=np.polynomial.legendre.leggauss(20)

##E

x=sym.Symbol("x",real=True)
y=sym.Symbol("y",real=True)

def GetLegendre(n,x,y):
    
    y=((x**2)-1)**n
    
    poly=sym.diff(y,x,n)/(2**n*np.math.factorial(n))
    
    return poly

Legendre=[]
N=20

for i in range(N+1):
    Poly=GetLegendre(i,x,y)
    Legendre.append(sym.lambdify([x],Poly,"numpy"))
    
##F

def GetCoeffs(f,p,N):
    
    coeffs=np.zeros(N+1)
    
    for n in range (N+1):
        
        I=np.sum(Weights*f(Roots)*p[n](Roots))
        
        cn=0.5*(2*n+1)*I
        
        coeffs[n]=cn
        
    return coeffs

C=GetCoeffs(func,Legendre,N)

##G

def Combinacion(x,p,N):
    
    Sum=0
    
    for k in range (N+1):
        
        Sum+=C[k]*p[k](x)
        
    return Sum

Aprox=Combinacion(X,Legendre,N)

##H

plt.title("Expansion en la base de Legendre Al grado n=20")
plt.scatter(X,Y,c="r",label="sgn(x)")
plt.plot(X,Aprox,c="b",label="Aproximacion En base de legendre")

plt.legend()
        
    
    
    



    



