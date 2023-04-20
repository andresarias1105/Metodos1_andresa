import numpy as np

def Matrix_mul(A,B):
    
    C=np.zeros([A.shape[0],B.shape[1]])
    
    if A.shape[1]==B.shape[0]:
        
        for i in range(A.shape[0]):
            
            for j in range(B.shape[1]):
                Sum=0
                for k in range(B.shape[0]):
                    
                    Sum+=A[i,k]*B[k,j]
                    
                C[i,j]=Sum
                
        
        return C
    
    else:
        return None
    
    
A=np.array([[1,0,0],[5,1,0],[-2,3,1]])
   
B=np.array([[4,-2,1],[0,3,7],[0,0,2]])




C=Matrix_mul(A,B)

print(C)