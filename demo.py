import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

#carrega modelo
clf = load('filename.joblib') 

entradas=[]
print("Digite o valor das features na mesma ordem do dataset disponibilizado utilizando virgula como separador e aperte enter. Quando terminar digite 'sair'. \n ")
var =input("Digite o valor das features: ")
while(var != 'sair'):
    
    var=var.split(',')
    
    for i in range(len(var)):
        if var[i]==' ':
            var.remove(' ')
        else:          
            var[i]=float(var[i])
    entradas.append(var)
    var =input("Digite o valor das features: ")

print(clf.predict(entradas))
