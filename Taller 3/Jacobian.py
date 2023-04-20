import numpy as np

def derivada_O4(f,x,y,z,variable,h=0.01):
    
    if variable==0:
    
       d=(1/(12*h)*f(x - 2*h,y,z)-2/(3*h)*f(x-h,y,z)+2/(3*h)*f(x+h,y,z)-1/(12*h)*f(x + 2*h,y,z))
       
    if variable==1:
    
       d=(1/(12*h)*f(x ,y-2*h,z)-2/(3*h)*f(x,y-h,z)+2/(3*h)*f(x,y+h,z)-1/(12*h)*f(x,y+ 2*h,z))
       
       
    if variable==2:
     
        d=(1/(12*h)*f(x,y,z-2*h)-2/(3*h)*f(x,y,z-h)+2/(3*h)*f(x,y,z+h)-1/(12*h)*f(x,y,z+ 2*h))
       
    
    
    return d




def Get_Jacobian_O4(G,r):
    
    dim=len(G)
    J=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            
            J[i,j]=derivada_O4(G[i],r[0],r[1],r[2],j)
            
    return J
            
G = (lambda x,y,z: 6*x - 2*np.cos(y*z) - 1., \
    lambda x,y,z: 9*y + np.sqrt(x**2 +np.sin(z) +1.06) + 0.9 , \
    lambda x,y,z: 60*z + 3*np.exp(-1*x*y)+ 10*np.pi +3.)
    
J_O4=Get_Jacobian_O4(G,[0.5,0.5,0.5])

print("Jacobiano orden 4:\n",J_O4)


def Get_Jacobian_O2(G,r,h=0.01):
    
    dim = len(G)
    
    J = np.zeros((dim,dim))
    
    for i in range(dim):
        J[i,0] = (G[i]( r[0] + h,r[1],r[2]) - G[i]( r[0] - h,r[1],r[2]))/(2*h)
        J[i,1] = (G[i]( r[0] ,r[1]+h,r[2]) - G[i]( r[0],r[1]-h,r[2]))/(2*h)
        J[i,2] = (G[i]( r[0],r[1],r[2]+h) - G[i]( r[0],r[1],r[2]-h))/(2*h)
        
    return J

J_O2=Get_Jacobian_O2(G,[0.5,0.5,0.5])

print("Jacobiano orden 2:\n",J_O2)