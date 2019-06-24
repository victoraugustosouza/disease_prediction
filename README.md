
# disease_prediction
Bulding a prediction model that can predict the presence of a disease in the patient through the dataset provided using SciKit-Learn

Requerimentos:
- Python 3.7 <br />
- DataSet disponibilizado, que deve estar na mesma pasta que os scripts.<br/><br/>
**Bibliotecas para Python:**
- Numpy
- Pandas
- Sklearn
-  joblib
  
**Como executar o código:**
-   Executar training.py normalmente, ele salvará o modelo treinado como modelo_treinado_salvo.joblib na mesma pasta.

-   O arquivo demo.py deve ser utilizados para testes, ele carregará o modelo e solicitará a entrada das valores das features. As features     devem ser inseridas utilizando virgula como separador e na mesma ordem do dataset produzido. Quando terminar de introduzir as features digite 'sair' e o programa dará a predição e terminará.

**Processo de Desenvolvimento:** <br/>
 O primeiro passo foi fazer o pré-processamento dos dados.</br> Carreguei os dados e procurei ver se haviam valores faltando(ou seja,nulos). Após isso,escolhi o modelo de aprendizagem supervisionada Support Vector Machine(SVM) para ser treinado.
Fiz data scaling nos dados visto que os valores grandes dominariam os pequenos,em especial no caso do algoritmo Support Vector Machine, e por ultimo separei os conjuntos de teste(30%) e treino(70%).<br/>

Na fase de treino utilizei GridSearch para otimizar parametrôs e 10-Fold Cross Validation para treinar o modelo SVM. Com o algotimo já treinado, utilizei o mesmo para prever os resultados do conjunto de teste usando a acurácia(número de acertos/total de amostras) como medidor de performance do modelo.
