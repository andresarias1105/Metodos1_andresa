import sympy as sym
sym.init_printing(use_unicode=False, wrap_line=False)
import numpy as np


A=sym.MatrixSymbol('A', 3, 3)
B=sym.MatrixSymbol('B', 3, 3)


J=np.zeros(3,dtype=object)

J[0]=sym.Matrix([[0,0,0],[0,0,-1],[0,1,0]])
J[1]=sym.Matrix([[0,0,1],[0,0,0],[-1,0,0]])
J[2]=sym.Matrix([[0,-1,0],[1,0,0],[0,0,0]])


def Conmutador(A,B):
    
    return A*B-B*A

Lie_bracket=Conmutador(A,B)


def Get_sign(i,j):
    x=Lie_bracket.subs([(A,J[i-1]),(B,J[j-1])]).doit()
    
   
    
    for l in range(1,4):
        
        if l!=i and l!=j:
            k=l
    
    if x==J[k-1]:
        
        return 1
    
    if x==-1*J[k-1]:
        
        return -1
    
    if x== sym.zeros(3,3):
        return 0
    
Levi_Civita=sym.zeros(3,3)

for i in range(1,4):
    for j in range(1,4):
        
        Levi_Civita[i-1,j-1]=Get_sign(i,j)
        
Levi_Civita
    
def Verificar(i,j):
    
    x=Lie_bracket.subs([(A,J[i-1]),(B,J[j-1])]).doit()
    
    for l in range(1,4):
        
        if l!=i and l!=j:
            k=l
    if x==J[k-1]*Levi_Civita[i-1,j-1]:
        
        print("Se cumple para i,j=",(i,j),"\n")
    else:
        return False
    
Verificar(1,2)##Permutacion par
Verificar(3,1)##Permutacion Impar
Verificar(2,2)##Permutacion i=j
     
    
    
    






        
       