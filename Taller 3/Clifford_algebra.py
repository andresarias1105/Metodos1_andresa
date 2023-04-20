import sympy as sym
sym.init_printing(use_unicode=False, wrap_line=False)
import numpy as np


A=sym.MatrixSymbol('A', 4, 4)
B=sym.MatrixSymbol('B', 4, 4)


gamma=np.zeros(4,dtype=object)

gamma[0]=sym.Matrix([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])
gamma[1]=sym.Matrix([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]])
gamma[2]=sym.Matrix([[0,0,0,-sym.I],[0,0,sym.I,0],[0,sym.I,0,0],[-sym.I,0,0,0]])
gamma[3]=sym.Matrix([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]])

def Clifford(A,B):
    
    return A*B+B*A

Product=Clifford(A,B)

metric=sym.diag(1,-1,-1,-1)

    
def Verificar(u,v):
    
    x=Product.subs([(A,gamma[u]),(B,gamma[v])]).doit()
    

        
       
    if x==2*metric[u,v]*sym.eye(4):
        print("Se cumple para u,v=",(u,v),"\n")
    else:
        return False
    
Verificar(1,2)##Permutacion par
Verificar(3,1)##Permutacion Impar
Verificar(2,2)
Verificar(3,3)
Verificar(1,1)
