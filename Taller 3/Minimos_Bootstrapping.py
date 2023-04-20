import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2

from scipy import integrate

import os
import os.path as path
import wget

from tqdm import tqdm
import corner
sample = np.array([0.974,0.950,0.932,1.104,1.038,0.920,0.935,0.907,0.810,0.915])
np.mean(sample)

def Resample(sample):
    
    resample_ = np.random.choice( sample, size=len(sample), replace=True )
    return resample_
    
np.mean(Resample(sample))
0.9522000000000002
def Bootstrapping(sample, N = int(1e5)):
    
    Mean_Dist = np.zeros( N )
    
    for i in tqdm(range(N)):
        resample_ = Resample(sample)
        Mean_Dist[i] = np.mean(resample_)
        
    return Mean_Dist
Mean_Dist = Bootstrapping(sample)

 
# Parametros de la distribucion
N = len(Mean_Dist)
xbar = np.mean(sample)
#xbar = np.percentile(Mean_Dist,50)
std = np.std(Mean_Dist)
print(xbar,std)

def Gaussian(x,mu,sigma): # Luego será el Likelihood
    
    return np.exp( -(x-mu)**2/(2*sigma**2) )/np.sqrt(2*np.pi*sigma**2)
x = np.linspace(np.min(sample),np.max(sample),100)
y = Gaussian(x,xbar,std)
plt.hist(Mean_Dist,bins=np.arange(np.min(sample),np.max(sample),0.01),density=True)
plt.plot(x,y,color='r')
plt.axvline(x=1,color='k')


#probabilidad de obtener usando el modelo de probabilidad
I = integrate.quad(Gaussian,1.,np.inf,args=(xbar,std))[0]
print(I)

# Usando los datos los eventos donde la media es igual o mayor a 1.
datasort = np.sort(Mean_Dist)

ii = np.where( datasort >= 1. )

EventosFavorables = np.sum( datasort[ii] )

frecuencia_relativa = EventosFavorables/N
print(frecuencia_relativa)

if not path.exists('Data'):
    os.mkdir('Data')
    
file = 'Data/Minimos.dat' 

url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/MinimosLineal.txt'
#url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/MinimosCuadratico.txt'

if not path.exists(file):
    Path_ = wget.download(url,file)
else:
    Path_ = file
data = np.loadtxt(Path_)
x = data[:,0]
y = data[:,1]
N = len(x)
sigma = np.random.uniform(0,0.2,N)
sigma
print(N)
20
plt.errorbar(x,y,yerr=sigma,fmt='o')


def GetFit(x,y,n=2):
    
    l = x.shape[0]
    b = y
    
    A = np.ones((l,n+1))
    
    for i in range(1,n+1):
        A[:,i] = x**i
        
    AT = np.dot(A.T,A)
    bT = np.dot(A.T,b)
    
    xsol = np.linalg.solve(AT,bT)
    
    return xsol
n = 1
param = GetFit(x,y,n)
param

def GetModel(x,p):
    
    y = 0
    for n in range(len(p)):
        y += p[n]*x**n
        
    return y
X = sym.Symbol('x',real=True)
GetModel(X,param)
_x = np.linspace(np.min(x),np.max(x),50)

_y = GetModel(_x,param)

plt.errorbar(x,y,yerr=sigma,fmt='o')
plt.plot(_x,_y,color='r')