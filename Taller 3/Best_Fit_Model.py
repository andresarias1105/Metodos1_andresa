import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



url ="https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Sigmoid.csv"
filename = "Data\Sigmoid.txt"
urllib.request.urlretrieve(url, filename)


Data = pd.read_csv(filename)
X = np.float64(Data.x)
Y = np.float64(Data.y)


def Model(theta,x):
    
    den=theta[1] + np.exp(-1*theta[2]*x)
    
    return theta[0]/den


def Cost_Function(X,Y,Model,theta):
     
    distance=(Y-Model(theta,X))
    sum_=np.sum(distance**2)
    
    return sum_

    



def GetGradient(G,r,h=1e-6):
    
    
    
    J = np.zeros(3)
    
  
    J[0] = (G(X,Y,Model,[r[0] + h,r[1],r[2]]) - G(X,Y,Model,[r[0] - h,r[1],r[2]]))/(2*h)
    J[1] = (G(X,Y,Model,[r[0],r[1]+h,r[2]]) - G(X,Y,Model,[r[0] ,r[1]-h,r[2]]))/(2*h)
    J[2] = (G(X,Y,Model,[r[0] ,r[1],r[2]+h]) - G(X,Y,Model,[r[0] ,r[1],r[2]-h]))/(2*h)
        
    return J


G=GetGradient(Cost_Function,[1,1,1])

def GetSolve(G,r,lr=5e-4,epochs=int(1e4),error=1e-4):
    
    d = 1
    it = 0
  
    
    while  d > error and it < epochs:
        
        
        grad= GetGradient(G,r)
   
        
        r -= lr*grad

        it += 1
        
        d=np.linalg.norm(lr*grad)
        
    return r

Parameters=GetSolve(Cost_Function,[1,1,1])



x_=np.linspace(-10, 10,100)
print(Parameters)
plt.scatter(X,Y,s=15,label="Data",c="r")
plt.plot(x_,Model(Parameters,x_),label="Best fit model",c="k")
plt.legend()




    
