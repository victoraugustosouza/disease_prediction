import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

#Pré-processamento de dados

#carregamento dos dados
data_frame = pd.read_csv('Mission_Prediction_Dataset.csv')

#mostrando primeiros registros
print(data_frame.head())

#número de linhas
print("Total de registros:"+str(data_frame.shape[0]))

#verificando se há valores nulos que devam ser tratados.
nulos_por_coluna=data_frame.isnull().sum()
print('Total de valores nulos no dataset:'+str(sum(nulos_por_coluna)))

#separando features do resultado
y=data_frame['column14'] #resultado
X=data_frame.iloc[:,0:13] #features

#scaling os dados 
feature_scaler = StandardScaler()  
X=feature_scaler.fit_transform(X.astype(float))  

#separando dados de Teste e de Treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42)

#Treinando Algoritmo utilizando Support Vector Machines e Grid Search para otimizar parametros 
clf = SVC(random_state=42)
parameters = {'kernel':('linear', 'rbf','poly'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
clf = GridSearchCV(clf, parameters,cv=10,scoring='accuracy') #validação cruzada durante o Grid-Search
clf.fit(X_train,y_train) #efetivamente ajusta o algoritmo

print('Parametros escolhidos:')
print(clf.best_params_)

#utilizando algoritmo treinado no conjunto de teste
y_pred=clf.predict(X_test)

#calculo a performance do seguinyr modo :numero de acertos/numero de amostras no conjunto de testes
print("Acertou:"+str(100*accuracy_score(y_test, y_pred))+'% das previsoes do conjunto de teste')
dump(clf, 'filename.joblib') 
