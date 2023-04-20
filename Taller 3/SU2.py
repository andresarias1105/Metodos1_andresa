import sympy as sym
sym.init_printing(use_unicode=False, wrap_line=False)
import numpy as np


A=sym.MatrixSymbol('A', 2, 2)
B=sym.MatrixSymbol('B', 2, 2)


sig=np.zeros(3,dtype=object)

sig[0]=sym.Matrix([[0,1],[1,0]])
sig[1]=sym.Matrix([[0,-sym.I],[sym.I,0]])
sig[2]=sym.Matrix([[1,0],[0,-1]])


def Conmutador(A,B):
    
    return A*B-B*A

Lie_bracket=Conmutador(A,B)


def Get_sign(i,j):
    x=Lie_bracket.subs([(A,sig[i-1]),(B,sig[j-1])]).doit()
    
   
    
    for l in range(1,4):
        
        if l!=i and l!=j:
            k=l
    
    if x==2*sym.I*sig[k-1]:
        
        return 1
    
    if x==-2*sym.I*sig[k-1]:
        
        return -1
    
    if x== sym.zeros(2,2):
        return 0
    
    
    
Levi_Civita=sym.zeros(3,3)

for i in range(1,4):
    for j in range(1,4):
        
        Levi_Civita[i-1,j-1]=Get_sign(i,j)
        

    
def Verificar(i,j):
    
    x=Lie_bracket.subs([(A,sig[i-1]),(B,sig[j-1])]).doit()
    
    for l in range(1,4):
        
        if l!=i and l!=j:
            k=l
    if x==sig[k-1]*Levi_Civita[i-1,j-1]*2*sym.I:
        
        print("Se cumple para i,j=",(i,j),"\n")
    else:
        return False
    
Verificar(1,2)##Permutacion par
Verificar(3,1)##Permutacion Impar
Verificar(2,2)##Permutacion i=j
    