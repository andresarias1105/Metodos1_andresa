import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sympy as sym
from tqdm import tqdm

n = 200
R = 1
x = np.linspace(-R,R,n+1)
y = np.linspace(-R,R,n+1)



def f(x,y,R=1):
    
    z = R**2 - x**2 - y**2
    
    if z <= 0.:
        return 0.
    else:
        return np.sqrt(z)
    
f = np.vectorize(f)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z)

def Integrate(X,Y,R,f):
    area=(X[0,1]-(X[0,0]))**2
    integral=0
    for i in tqdm(range(n)):
        for j in range(n):
            
            prom=f(X[i,j],Y[i,j])+f(X[i+1,j],Y[i+1,j])+f(X[i,j+1],Y[i,j+1])+f(X[i+1,j+1],Y[i+1,j+1])
            
            prom=prom/4
            
            integral+=prom*area
    
    return integral
            
            
Volumen_num=Integrate(X,Y,R,f)
Volumen_exacto=2/3 *np.pi*R**3

print("El Volumen de la semiesfera aproximado numericamente es:",Volumen_num)
                


