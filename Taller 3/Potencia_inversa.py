import numpy as np

A=np.array([[1,2,-1],[1,0,1],[4,-4,5]])

np.linalg.eig(A)

def EigValue(A,c=0,itmax=1000,tolerancia=1e-16):
    
    n = A.shape[0]
    
    A=np.linalg.inv(A)
    
    v0 = np.zeros(n)
    
    v0[c] = 1.
    
    lambda1 = 0.
    
    for k in range(itmax):
        
        v1 = np.dot(A,v0)
        v1 = v1/np.linalg.norm(v1) # Normaliza
        
        v2 = np.dot(A,v1)
        v2 = v2/np.linalg.norm(v1)
        
        lambda0 = lambda1
        lambda1 = v2[0]/v1[0]
        
        v0 = v2
        
        if np.abs(lambda0 - lambda1) <= tolerancia:
            break
            
    return lambda1,v1

value1,vector1 = EigValue(A)

print("El valor propio mas pequeño y su vector asociado son:")
print(value1,vector1)
