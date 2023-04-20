import numpy as np
import math as math


def Mat_Rot(p,q,dim,theta):
    
    R=np.zeros([dim,dim])
    
    for i in range(dim):
        R[i,i]=1
    
    R[p,q]=-1*np.sin(theta)
    R[q,p]=1*np.sin(theta)
    R[p,p]=np.cos(theta)
    R[q,q]=np.cos(theta)     
    
    return R

def Get_max_pos(A):
    
    B=A.copy()
    
    B=np.abs(B)
    
    
    
    for k in range(B.shape[0]):
      B[k,k]=0
    for pi in range(B.shape[0]):
        for qi in range(pi+1,B.shape[0]):
            if B[pi,qi]==np.max(B):
               p=pi
               q=qi
    
    return p,q
    


         

def Jacobi_Diagonal(A,tolerance=1e-14,itmax=1000):
    
    dim=A.shape[0]
    R=np.identity(dim)
    i,j=Get_max_pos(A)
    it=0
    B=A.copy()
    while np.max(np.abs(B))>tolerance and it<itmax:
        
      if A[i,i]==A[j,j]:
          theta=0.25*np.pi
      else:
          theta=0.5*np.arctan(2*A[i,j]/(A[i,i]-A[j,j]))
          
      Ri=Mat_Rot(i,j,dim,theta)
      

      R=R@Ri
      
      A=np.transpose(Ri)@A@Ri
      
      B=A.copy()
      
      B=np.abs(B)
      

      
      for k in range(dim):
        B[k,k]=0
        
 
            
      for pi in range(dim):
          for qi in range(pi+1,dim):
              if math.isclose(B[pi,qi],np.max(np.abs(B))):
                  
                 i=pi
                 
                 j=qi
     
      
     

     
      it+=1
      
      
     
    
    E_val=np.zeros(dim)

    for z in range(dim):
      E_val[z]=A[z,z]
    return E_val,R
        
        
A=np.array([[4,1,1],[1,3,2],[1,2,5]])

EigVectors,EigValues=Jacobi_Diagonal(A)

NumpyVectors,NumpyValues=np.linalg.eig(A)

print("Valores Propios:\n")
print("Jacobi:",(EigVectors),"Numpy:",NumpyVectors,"\n")

print("Vectores Propios:\n")
print("Jacobi:",EigValues,"\n","Numpy:",NumpyValues)
