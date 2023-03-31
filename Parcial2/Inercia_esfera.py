import numpy as np

from tqdm import tqdm

def CreateSphere(N,R=1):
    
    Points = np.zeros((N,3))
        
    for i in tqdm(range(N)):
        
        phi = np.random.uniform(0,2*np.pi)
        u = np.random.rand()
        r = R*u**(1/3)
        costheta = np.random.uniform(-1,1)
        theta = np.arccos(costheta)
        
        Points[i] = [r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)]
        
    return Points

N=1e6

sphere=CreateSphere(int(N))
def Iab(a,b):
    
    return a**2+b**2




def ixy(a,b):
    
    return -1*a*b
muestra=np.zeros([sphere.shape[0],4])

for i in tqdm(range(sphere.shape[0])):
    muestra[i][0]=Iab(sphere[i,1],sphere[i,2])
    muestra[i][1]=Iab(sphere[i,0],sphere[i,2])
    muestra[i][2]=Iab(sphere[i,0],sphere[i,1])
    muestra[i][3]=ixy(sphere[i,0],sphere[i,1])
    
    
Ixx=np.average(muestra[:,0])
Iyy=np.average(muestra[:,1])
Izz=np.average(muestra[:,2])
Ixy=np.average(muestra[:,3])
print("\n")
print("Ixx=", Ixx)
print("Iyy=", Iyy)
print("Izz=", Izz)
print("Ixy=", Ixy)

"""para 1e6 puntos el error es de el orden de 1e-3,

consistente con el valor esperado de 1/sqrt(N)=1/sqrt(1e6)=1e-3"""

"""La esfera es simetrica sobre la rotacion , ya que
 los momentos de inercia son invariables para rotaciones sobre diferentes 
 ejes. Adicionalmente el valor del producto de inercia es cercano a cero
Lo que indica una simetria que produce un centroide nulo. """





