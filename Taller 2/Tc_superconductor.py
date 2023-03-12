import numpy as np
from scipy import integrate

def funcion(x,T,deltaT):
    
    t=(np.sqrt(x**2+deltaT**2))
    
    f=np.tanh(t*(300./(2*T)))
    
    
    return 0.5*(f/t)
       


Roots,Weights=np.polynomial.legendre.leggauss(20)

def Integrate(Rootsi,Weightsi,f,a,b,T,deltaT=0):
    t=0.5*((b-a)*Rootsi+a+b)

    integral=0.5*(b-a)*np.sum(Weightsi*f(t,T,deltaT))
    
    return integral


def Critical_temp(rango):
    
    Tn=np.linspace(1,rango,100000)
    N0V=0.3
    for i in range(len(Tn)):
      Tc=Tn[i]

      I=Integrate(Roots,Weights,funcion,-1,1,Tc)
      
      
      if  np.abs(I-(1/(N0V))) < 1e-5: 
         return Tc
      
        
Tc=Critical_temp(21)   
I=Integrate(Roots,Weights,funcion,-1,1,Tc,0)
Int=integrate.fixed_quad(funcion,-1,1,args=(Tc,0),n=20)
print(Tc)


  
    