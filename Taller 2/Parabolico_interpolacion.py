import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
url ="https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Parabolico.csv"
filename = "Data\Parabolico.txt"
urllib.request.urlretrieve(url, filename)


Data = pd.read_csv(filename)
X = np.float64(Data.X)
Y = np.float64(Data.Y)



def Lagrange(x,xi,j):
    
    prod = 1.0
    n = len(xi)
    
    for i in range(n):
        if i != j:
            prod *= (x - xi[i])/(xi[j]-xi[i])
            
    return prod
def Interpolate(x,xi,yi):
    
    Sum = 0.
    n = len(xi)
    
    for j in range(n):
        Sum += yi[j]*Lagrange(x,xi,j)
        
    return Sum

x0 = np.linspace(X[0],X[-1],1000)
trayectoria = Interpolate(x0,X,Y)
plt.scatter(X,Y,marker='o',color='r')
plt.plot(x0,trayectoria,color='k')

x=sym.Symbol("x",real=True)

fun_trayectoria=(Interpolate(x,X,Y))
fun_trayectoria=sym.simplify(fun_trayectoria)
fun_trayectoria=sym.expand(fun_trayectoria)

Coeficiente_lineal=fun_trayectoria.args[0].args[0]
Coeficiente_Cuadratico=fun_trayectoria.args[1].args[0]

##La Ec.de trayectoria es: y=h+xtan(theta)+gx**2/2vo**2*cos(theta)**2
## Entonces el coeficiente lineal es igual tan(theta)
## El coeficiente cudratico es igual a g/2vo**2*cos(theta)**2

g=-9.8

theta=sym.atan(Coeficiente_lineal)
V0=sym.sqrt(g/(2*Coeficiente_Cuadratico*sym.cos(theta)**2))
theta=theta*180/np.pi

V=np.array([V0,theta])

print("La funcion de trayectoria es:",fun_trayectoria)
print("El angulo es:",theta,"grados")
print("La rapidez inical es:",V0,"m/s")
