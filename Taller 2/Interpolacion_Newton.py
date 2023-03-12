# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:01:51 2023

@author: THINKBOOK
"""

import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import pandas as pd
import sympy as sym



url ="https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/InterpolacionNewtonNoequi.csv"
filename = "Data\Interpolacion_Newton.txt"
urllib.request.urlretrieve(url, filename)


Data = pd.read_csv(filename)
X = np.float64(Data.X)
Y = np.float64(Data.Y)

def Diferencia_dividida(X, Y):
    
    
    Diff = np.zeros([len(X), len(X)])
   
    Diff[:,0] = Y
    
    for j in range(1,len(X)):
        for i in range(len(X)-j):
            Diff[i][j] = (Diff[i+1][j-1] - Diff[i][j-1]) / (X[i+j]-X[i])
           
            
    return Diff

def Interpolar(Diff, X, xn):
    
    n = len(X) - 1 
    p = Diff[n]
    for k in range(1,n+1):
        p = Diff[n-k] + (xn -X[n-k])*p
    return p

# get the divided difference coef
Coeficientes = Diferencia_dividida(X, Y)[0, :]
xn=np.linspace(X[0],X[-1],100)

yn = Interpolar(Coeficientes, X, xn)

x=sym.Symbol("x",real=True)


polinomio_interpolador=Interpolar(Coeficientes,X,x)
polinomio_interpolador=sym.simplify(polinomio_interpolador)

print("El polinomio que interpola los puntos es:",polinomio_interpolador)

plt.scatter(X, Y,c="r",label="Data")
plt.plot(xn, yn,c="b",label="Interpolacion")

plt.legend()