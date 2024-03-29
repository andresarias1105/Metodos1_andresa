import numpy as np
from scipy import integrate

def funcion(x,T,deltaT):
    
    t=(np.sqrt(x**2+deltaT**2))
    
    f=np.tanh(t*(300./(2*T)))
    
    
    return 0.5*(f/t)
       


Roots,Weights=np.polynomial.legendre.leggauss(50)

def Integrate(Rootsi,Weightsi,f,a,b,T,deltaT=0):
    t=0.5*((b-a)*Rootsi+a+b)

    integral=0.5*(b-a)*np.sum(Weightsi*f(t,T,deltaT))
    
    return integral


def Critical_temp(rango,paso):
    
    Tn=np.arange(1,rango,paso)
    N0V=0.3
    for i in range(len(Tn)):
      Tc=Tn[i]

      I=Integrate(Roots,Weights,funcion,-1,1,Tc)
      
      
      if  np.abs(I-(1/(N0V))) < paso: 
         return Tc
      
        
Tc=Critical_temp(20,1e-4)   
I=Integrate(Roots,Weights,funcion,-1,1,Tc,0)
Int=integrate.fixed_quad(funcion,-1,1,args=(Tc,0),n=20)
print("la temperatura critica es aproximadamente: ",round(Tc,4))


  
    