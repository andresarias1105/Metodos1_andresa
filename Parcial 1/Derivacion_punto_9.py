import sympy as sym

x = sym.Symbol("x",real=True)
h=sym.Symbol("h",real=True)

X=[-2*h,-1*h,0*h,1*h,2*h]

def Lagrange(x,xi,j):
    
    prod = 1
    n = len(xi)
    
    for i in range(n):
        if i != j:
            prod *= (x - xi[i])/(xi[j]-xi[i])
            
    return prod
def Interpolate(x,xi,yi):
    
    Sum = 0.
    n = len(xi)
    
    for j in range(n):
        Sum += yi[j]*Lagrange(x,xi,j)
        
    return Sum


def GetCoefficients(x,p,X):
    Coefficients = []
   
    for i in range(len(X)):
        Li=Lagrange(x,X,i)
        
        dLi=sym.diff(Li,x,1)
        
        C=dLi.subs(x,X[p])
        
        Coefficients.append(C)
    
    return Coefficients
        
print(GetCoefficients(x,2,X))

"""
I) El orden de aproximacion para la dreivada progresiva, calculada con 
    El polinomio interploacion de lagrange de n+1 puntos, es de orden n.
    En el ejercicio se toman 5 puntos, por lo que la aproximacion es de orden 4
    
    """