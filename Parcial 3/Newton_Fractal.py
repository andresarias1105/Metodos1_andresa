import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from tqdm import tqdm

##a

x=sym.Symbol("x",real=True)
y=sym.Symbol("y",real=True)

##b

z=x+sym.I*y


##c

def Function(z):
    
    return z**3 -1

f=Function(z)

##d

F= [sym.re(f),sym.im(f)]

##e

def Jacobian(F):
    
    J=sym.Matrix([[sym.diff(F[0],x),sym.diff(F[0],y)],[sym.diff(F[1],x),sym.diff(F[1],y)]])
    
 
    return J


J=Jacobian(F)
print(J)

##d
Fn = sym.lambdify([x,y],F,"numpy")
Jn = sym.lambdify([x,y],J,"numpy")

##G
def Newton(z0,Fn,Jn,tol=1e-7,itmax=1000):
    
    error=1
    it=0
    
    while error>tol and it<itmax:
        
        Jinv=np.linalg.inv(Jn(z0[0],z0[1]))
        
        zn=z0-np.dot(Jinv,Fn(z0[0],z0[1]))
        
        error=np.linalg.norm(zn-z0)
        
        z0=zn
    
        

    return z0

##H

Root=Newton([0.5,0.5],Fn,Jn)
print(Root)

##I

N=1000

x_= np.linspace(-1, 1, N)
y_= np.linspace(-1, 1, N)

Allroots=[[-0.5,0.8660254],[-0.5,-0.8660254],[1,0]]

##J
Fractal = np.zeros((N,N), np.int64)

for i in tqdm(range(N)):
    for j in range(N):
        
        Rooti=Newton([x_[i],y_[j]],Fn,Jn)
        
        if np.allclose(Rooti,Allroots[0]):
            
            Fractal[i,j]=20
            
        if np.allclose(Rooti,Allroots[1]):
             
             Fractal[i,j]=100
        if np.allclose(Rooti,Allroots[2]):
              
             Fractal[i,j]=255
             
##K
plt.imshow(Fractal, cmap="coolwarm" ,extent=[-1,1,-1,1])

        
        
        
