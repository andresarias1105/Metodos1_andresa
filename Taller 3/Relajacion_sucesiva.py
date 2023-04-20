import numpy as np

import matplotlib.pyplot as plt

def Relajacion_sucesiva(A, b, omega, x0, tolerancia):
    
    step = 0
    phi = x0.copy()
    residuo = 1
    while residuo > tolerancia:
        
     
        for i in range(A.shape[0]):
            sigma = 0
            pn0=phi.copy()
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i, j] * phi[j]
                    
            phi[i] = (1 - omega) * phi[i] + (omega / A[i, i]) * (b[i] - sigma)
        residuo = np.abs((phi[j]-pn0[j])/phi[j])
        step += 1
        
        """
        print("Step {} Residuo: {:10.6g}".format(step, residuo))
        print( "Solucion:", phi)
        """
    
         
              
    return phi,step

crit_convergencia = 1e-8
omega = 1

A = np.array([[3,-1,-1],[-1.,3.,1.],[2,1,4]])
b = np.array([1.,3.,7.])

vector_init = np.zeros(b.shape[0])

sol=Relajacion_sucesiva(A,b,omega,vector_init,crit_convergencia)


w_=np.linspace(1,2,50)
steps=np.zeros([len(w_)])

for it in range(len(w_)):
    
    steps[it]=Relajacion_sucesiva(A,b,w_[it],vector_init,crit_convergencia)[1]
    
print("La Solucion del sistema es:",sol[0])


plt.plot(w_,steps)
plt.xlabel("$\omega$")
plt.ylabel("Steps")
plt.yscale("log")
"""El metodo converge en el menor numero de iteraciones para omega=1,
y empieza a diverger despues de 1,4<omega<1,5"""
    
    
    
    

