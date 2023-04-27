import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def GetSol(A,b):
    A_=A.T
    AT=np.dot(A_,A)
    BT=np.dot(A_,b)
    
    sol=np.linalg.solve(AT,BT)
    
    return sol

A = np.array([[2.,-1.],[1.,2.],[1.,1.]])
b = np.array([2.,1.,4.])
def GetY(A,b,x,i):
    return (b[i]-A[i,0]*x/A[i,1])
x = np.linspace(-10,10,50)

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(3):
    ax.plot(x,GetY(A,b,x,i))
    
sol=GetSol(A,b)



ax.scatter(sol[0],sol[1])

""" No hay punto de interseccion entre las tres lineas,
dado a que el sistema lineal es inconsistente.El punto encontrado
es el que minimiza la distancia euclidiana a las tres lineas"""
    
def Get_distance(A,b,x,y):
    
    X=np.array([x,y])
    
    point=np.dot(A,X)
    
    return np.linalg.norm(point-b)

 
h=0.01
x = np.arange(-5,5,h)
y = np.arange(-5,5,h)
X,Y = np.meshgrid(x,y)

d=np.zeros((len(x),len(y)))

for i in tqdm(range(len(x))):
    for j in range(len(y)):
        
        d[i,j]=Get_distance(A,b,X[i,j],Y[i,j])

fig2 = plt.figure()
ax1 = fig2.add_subplot(111,projection='3d')
ax1.plot_surface(X,Y,d,cmap="coolwarm")
ax1.set_title(label=" Distancia ||Ax∗ −b||")
zsol= np.linalg.norm(np.dot(A,sol)-b)
ax1.scatter(sol[0],sol[1],zsol,color="r",label="Solucion con MC")

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("D")
ax1.legend()
ax1.view_init(elev=7, azim=45)
print(zsol,np.min(d))


"""Ambas soluciones son igules en sus primeras
6 cifras"""

        


